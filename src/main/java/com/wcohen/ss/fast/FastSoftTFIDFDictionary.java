package com.wcohen.ss.fast;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;

import com.wcohen.ss.api.Token;
import com.wcohen.ss.lookup.FastLookup;
import com.wcohen.ss.lookup.LookupResult;

/**
 * High-performance SoftTFIDFDictionary with:
 * - parallel freeze
 * - precomputed token-token similarities
 * - cached prepared vectors for dictionary entries
 * - compact int candidate sets
 */
public class FastSoftTFIDFDictionary implements FastLookup {

    private static final int DEFAULT_WINDOW_SIZE = 100;
    private static final double DEFAULT_MIN_TOKEN_SIMILARITY = 0.9;
    private static final int DEFAULT_MAX_INVERTED_INDEX_SIZE = 0;

    private final FastSimpleTokenizer tokenizer;
    private final FastTFIDF tfidfDistance;
    private final FastSoftTFIDF softTFIDFDistance;
    private final FastJaroWinkler jaroWinklerDistance;

    private int windowSize;
    private int maxInvertedIndexSize;
    private int maxCandidatesToScore;
    private boolean approximateRetrievalEnabled;
    private final double minTokenSimilarity;

    private final HashMap<String, ArrayList<String>> valueMap = new HashMap<>();
    private boolean frozen = false;

    private double[] maxTFIDFScore;
    private int[][] similarTokenIds;
    private double[][] similarTokenSims;
    private int[][] invertedIndexDocIds;
    private Token[] allTokens;
    private int numTokens;

    private String[] dictStrings;
    private FastBagOfTokens[] dictBags;
    private ArrayList<String>[] valuesByDoc;

    private List<LookupResult> result = new ArrayList<>();
    private double lookupTime;

    private static final ThreadLocal<LookupScratch> LOOKUP_SCRATCH = ThreadLocal.withInitial(LookupScratch::new);

    private static final class LookupScratch {
        double[] upperBounds = new double[8192];
        boolean[] ubSet = new boolean[8192];
        int[] ubIds = new int[4096];
        int ubCount = 0;

        boolean[] candidateFlags = new boolean[8192];
        int[] candidateIds = new int[4096];
        double[] candidateBounds = new double[8192];
        int candidateCount = 0;

        boolean[] sUsed = new boolean[64];
        boolean[] tUsed = new boolean[64];
    }

    public FastSoftTFIDFDictionary() {
        this(new FastSimpleTokenizer(false, true), DEFAULT_MIN_TOKEN_SIMILARITY, DEFAULT_WINDOW_SIZE, DEFAULT_MAX_INVERTED_INDEX_SIZE);
    }

    public FastSoftTFIDFDictionary(FastSimpleTokenizer tokenizer, double minTokenSimilarity, int windowSize, int maxInvertedIndexSize) {
        this.tokenizer = tokenizer;
        this.minTokenSimilarity = minTokenSimilarity;
        this.windowSize = windowSize;
        this.maxInvertedIndexSize = maxInvertedIndexSize;
        this.maxCandidatesToScore = 0;
        this.approximateRetrievalEnabled = false;
        this.tfidfDistance = new FastTFIDF(tokenizer);
        this.jaroWinklerDistance = new FastJaroWinkler();
        this.softTFIDFDistance = new FastSoftTFIDF(tokenizer, jaroWinklerDistance, minTokenSimilarity);
    }

    public void setWindowSize(int w) { this.windowSize = w; }
    public void setMaxInvertedIndexSize(int m) { this.maxInvertedIndexSize = m; }
    public void setMaxCandidatesToScore(int m) { this.maxCandidatesToScore = Math.max(0, m); }
    public void setApproximateRetrievalEnabled(boolean enabled) { this.approximateRetrievalEnabled = enabled; }
    public boolean isApproximateRetrievalEnabled() { return approximateRetrievalEnabled; }

    /**
     * Convenience API for opt-in approximate retrieval mode.
     * When enabled, lookup() scores only the highest-bound candidates.
     */
    public void enableApproximateRetrieval(int maxCandidatesToScore) {
        this.approximateRetrievalEnabled = true;
        this.maxCandidatesToScore = Math.max(1, maxCandidatesToScore);
    }

    public void disableApproximateRetrieval() {
        this.approximateRetrievalEnabled = false;
    }
    public double getLookupTime() { return lookupTime; }

    public void put(String string, Object value) {
        if (frozen) throw new IllegalStateException("can't add to frozen dictionary");
        valueMap.computeIfAbsent(string, k -> new ArrayList<>(2)).add(String.valueOf(value));
    }

    public void loadAliases(File file) throws IOException {
        try (BufferedReader in = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = in.readLine()) != null) {
                String[] parts = line.split("\\t");
                for (int j = 1; j < parts.length; j++) put(parts[j], parts[0]);
            }
        }
    }

    public void freeze() {
        if (frozen) return;

        dictStrings = valueMap.keySet().toArray(new String[0]);
        tfidfDistance.train(dictStrings);
        softTFIDFDistance.train(dictStrings);

        int maxId = tokenizer.maxTokenIndex();
        maxTFIDFScore = new double[maxId + 1];

        @SuppressWarnings("unchecked")
        ArrayList<Integer>[] invDoc = new ArrayList[maxId + 1];

        dictBags = new FastBagOfTokens[dictStrings.length];
        valuesByDoc = new ArrayList[dictStrings.length];

        for (int docId = 0; docId < dictStrings.length; docId++) {
            String s = dictStrings[docId];
            FastBagOfTokens vec = tfidfDistance.prepareVector(s);
            dictBags[docId] = softTFIDFDistance.prepareVector(s);
            valuesByDoc[docId] = valueMap.get(s);

            int[] ids = vec.getTokenIds();
            double[] w = vec.getWeights();
            for (int i = 0; i < ids.length; i++) {
                int id = ids[i];
                if (w[i] > maxTFIDFScore[id]) maxTFIDFScore[id] = w[i];
                if (invDoc[id] == null) invDoc[id] = new ArrayList<>();
                invDoc[id].add(docId);
            }
        }

        invertedIndexDocIds = new int[maxId + 1][];
        for (int i = 0; i <= maxId; i++) {
            if (invDoc[i] == null) {
                invertedIndexDocIds[i] = new int[0];
            } else {
                int[] arr = new int[invDoc[i].size()];
                for (int j = 0; j < arr.length; j++) arr[j] = invDoc[i].get(j);
                invertedIndexDocIds[i] = arr;
            }
        }

        allTokens = new Token[Math.max(1, maxId)];
        numTokens = 0;
        Iterator<Token> it = tokenizer.tokenIterator();
        while (it.hasNext()) allTokens[numTokens++] = it.next();
        allTokens = Arrays.copyOf(allTokens, numTokens);
        Arrays.sort(allTokens, Comparator.comparing(Token::getValue));

        similarTokenIds = new int[maxId + 1][];
        similarTokenSims = new double[maxId + 1][];
        computeSimilarTokensParallel();

        frozen = true;
    }

    private void computeSimilarTokensParallel() {
        if (numTokens < 1000 || Runtime.getRuntime().availableProcessors() <= 1) {
            for (int i = 0; i < numTokens; i++) fillSimilarForTokenAt(i);
            return;
        }

        ForkJoinPool pool = ForkJoinPool.commonPool();
        CountDownLatch latch = new CountDownLatch(numTokens);
        for (int i = 0; i < numTokens; i++) {
            final int idx = i;
            pool.submit(() -> {
                try {
                    fillSimilarForTokenAt(idx);
                } finally {
                    latch.countDown();
                }
            });
        }
        try {
            latch.await();
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(ex);
        }
    }

    private void fillSimilarForTokenAt(int pos) {
        String s = allTokens[pos].getValue();
        int lo = Math.max(0, pos - windowSize);
        int hi = Math.min(numTokens, pos + windowSize);

        int[] idBuf = new int[Math.max(1, hi - lo)];
        double[] simBuf = new double[Math.max(1, hi - lo)];
        int c = 0;

        for (int j = lo; j < hi; j++) {
            if (j == pos) continue;
            Token tj = allTokens[j];
            double sim = jaroWinklerDistance.scoreStrings(s, tj.getValue());
            if (sim >= minTokenSimilarity) {
                idBuf[c] = tj.getIndex();
                simBuf[c] = sim;
                c++;
            }
        }

        int tokId = allTokens[pos].getIndex();
        similarTokenIds[tokId] = Arrays.copyOf(idBuf, c);
        similarTokenSims[tokId] = Arrays.copyOf(simBuf, c);
    }

    @Override
    public int lookup(double minScore, String toFind) {
        if (!frozen) freeze();
        long start = System.nanoTime();

        FastBagOfTokens q = tfidfDistance.prepareVector(toFind);
        int[] qIds = q.getTokenIds();
        double[] qW = q.getWeights();

        LookupScratch sc = LOOKUP_SCRATCH.get();
        int maxId = tokenizer.maxTokenIndex();
        if (sc.upperBounds.length <= maxId) {
            sc.upperBounds = new double[maxId + 1];
            sc.ubSet = new boolean[maxId + 1];
            sc.ubIds = new int[maxId + 1];
        }
        if (sc.candidateFlags.length < dictStrings.length) {
            sc.candidateFlags = new boolean[dictStrings.length];
            sc.candidateIds = new int[Math.max(4096, dictStrings.length)];
            sc.candidateBounds = new double[dictStrings.length];
        } else if (sc.candidateIds.length < dictStrings.length) {
            sc.candidateIds = new int[dictStrings.length];
            sc.candidateBounds = new double[dictStrings.length];
        }

        sc.ubCount = 0;
        sc.candidateCount = 0;

        for (int i = 0; i < qIds.length; i++) {
            int qId = qIds[i];
            if (qId >= maxTFIDFScore.length) continue;
            double qw = qW[i];
            if (maxTFIDFScore[qId] > 0) {
                addUpperBound(sc, qId, qw * maxTFIDFScore[qId]);

                int[] sims = qId < similarTokenIds.length ? similarTokenIds[qId] : null;
                double[] simVals = qId < similarTokenSims.length ? similarTokenSims[qId] : null;
                if (sims != null && simVals != null) {
                    for (int k = 0; k < sims.length; k++) {
                        int sid = sims[k];
                        addUpperBound(sc, sid, qw * maxTFIDFScore[sid] * simVals[k]);
                    }
                }
            }
        }

        int[] sorted = Arrays.copyOf(sc.ubIds, sc.ubCount);
        for (int i = 1; i < sorted.length; i++) {
            int key = sorted[i];
            double kv = sc.upperBounds[key];
            int j = i - 1;
            while (j >= 0 && sc.upperBounds[sorted[j]] > kv) {
                sorted[j + 1] = sorted[j];
                j--;
            }
            sorted[j + 1] = key;
        }

        double total = 0.0;
        boolean useCandidateBounds = approximateRetrievalEnabled && maxCandidatesToScore > 0;
        for (int id : sorted) {
            total += sc.upperBounds[id];
            if (total >= minScore) {
                int[] ii = invertedIndexDocIds[id];
                if (maxInvertedIndexSize <= 0 || ii.length < maxInvertedIndexSize) {
                    double tokenBound = sc.upperBounds[id];
                    for (int docId : ii) {
                        if (!sc.candidateFlags[docId]) {
                            sc.candidateFlags[docId] = true;
                            sc.candidateIds[sc.candidateCount++] = docId;
                            if (useCandidateBounds) sc.candidateBounds[docId] = tokenBound;
                        } else if (useCandidateBounds) {
                            sc.candidateBounds[docId] += tokenBound;
                        }
                    }
                }
            }
        }

        for (int i = 0; i < sc.ubCount; i++) {
            int id = sc.ubIds[i];
            sc.upperBounds[id] = 0.0;
            sc.ubSet[id] = false;
        }

        result = new ArrayList<>(sc.candidateCount);
        FastBagOfTokens qBag = softTFIDFDistance.prepareVector(toFind);
        HashMap<Long, Double> tokenSimCache = new HashMap<>(4096);

        int scoreCount = sc.candidateCount;
        int[] scoreDocIds = sc.candidateIds;
        if (useCandidateBounds && sc.candidateCount > maxCandidatesToScore) {
            int k = maxCandidatesToScore;
            int[] topIds = new int[k];
            double[] topBounds = new double[k];
            int topN = 0;

            for (int i = 0; i < sc.candidateCount; i++) {
                int docId = sc.candidateIds[i];
                double b = sc.candidateBounds[docId];

                if (topN < k) {
                    int p = topN;
                    while (p > 0 && topBounds[p - 1] < b) {
                        topBounds[p] = topBounds[p - 1];
                        topIds[p] = topIds[p - 1];
                        p--;
                    }
                    topBounds[p] = b;
                    topIds[p] = docId;
                    topN++;
                } else if (b > topBounds[k - 1]) {
                    int p = k - 1;
                    while (p > 0 && topBounds[p - 1] < b) {
                        topBounds[p] = topBounds[p - 1];
                        topIds[p] = topIds[p - 1];
                        p--;
                    }
                    topBounds[p] = b;
                    topIds[p] = docId;
                }
            }

            scoreCount = topN;
            scoreDocIds = topIds;
        }

        for (int i = 0; i < scoreCount; i++) {
            int docId = scoreDocIds[i];
            sc.candidateFlags[docId] = false;
            if (useCandidateBounds) sc.candidateBounds[docId] = 0.0;

            double d = scoreBagsCached(qBag, dictBags[docId], tokenSimCache, sc);
            if (d >= minScore) {
                ArrayList<String> vals = valuesByDoc[docId];
                if (vals != null) {
                    String found = dictStrings[docId];
                    for (String v : vals) result.add(new LookupResult(found, v, d));
                }
            }
        }

        if (scoreCount != sc.candidateCount) {
            for (int i = 0; i < sc.candidateCount; i++) {
                int docId = sc.candidateIds[i];
                sc.candidateFlags[docId] = false;
                if (useCandidateBounds) sc.candidateBounds[docId] = 0.0;
            }
        }

        result.sort(null);
        lookupTime = (System.nanoTime() - start) / 1_000_000_000.0;
        return result.size();
    }

    public int slowLookup(double minScore, String toFind) {
        if (!frozen) freeze();
        long start = System.nanoTime();
        result = new ArrayList<>();

        FastBagOfTokens q = softTFIDFDistance.prepareVector(toFind);
        for (int docId = 0; docId < dictStrings.length; docId++) {
            double d = softTFIDFDistance.scoreBags(q, dictBags[docId]);
            if (d >= minScore) {
                ArrayList<String> vals = valuesByDoc[docId];
                if (vals != null) {
                    String found = dictStrings[docId];
                    for (String v : vals) result.add(new LookupResult(found, v, d));
                }
            }
        }

        result.sort(null);
        lookupTime = (System.nanoTime() - start) / 1_000_000_000.0;
        return result.size();
    }

    private void addUpperBound(LookupScratch sc, int tokenId, double bound) {
        if (!sc.ubSet[tokenId]) {
            sc.ubSet[tokenId] = true;
            sc.upperBounds[tokenId] = bound;
            sc.ubIds[sc.ubCount++] = tokenId;
        } else if (bound > sc.upperBounds[tokenId]) {
            sc.upperBounds[tokenId] = bound;
        }
    }

    private double scoreBagsCached(FastBagOfTokens sBag, FastBagOfTokens tBag,
                                   HashMap<Long, Double> tokenSimCache,
                                   LookupScratch sc) {
        int sSize = sBag.size();
        int tSize = tBag.size();
        if (sSize == 0 || tSize == 0) return 0.0;

        int[] sIds = sBag.getTokenIds();
        int[] tIds = tBag.getTokenIds();
        String[] sVals = sBag.getTokenValues();
        String[] tVals = tBag.getTokenValues();
        double[] sWeights = sBag.getWeights();
        double[] tWeights = tBag.getWeights();

        if (sc.sUsed.length < sSize) sc.sUsed = new boolean[sSize];
        if (sc.tUsed.length < tSize) sc.tUsed = new boolean[tSize];
        for (int i = 0; i < sSize; i++) sc.sUsed[i] = false;
        for (int j = 0; j < tSize; j++) sc.tUsed[j] = false;

        double sim = 0.0;
        int remaining = Math.min(sSize, tSize);

        while (remaining > 0) {
            double best = -1.0;
            int bestI = -1;
            int bestJ = -1;

            for (int i = 0; i < sSize; i++) {
                if (sc.sUsed[i]) continue;
                String sVal = sVals[i];
                double sw = sWeights[i];
                int sid = sIds[i];

                for (int j = 0; j < tSize; j++) {
                    if (sc.tUsed[j]) continue;
                    int tid = tIds[j];

                    double tokenSim;
                    if (sid == tid || sVal.equals(tVals[j])) {
                        tokenSim = 1.0;
                    } else {
                        long key = (((long) sid) << 32) ^ (tid & 0xffffffffL);
                        Double cached = tokenSimCache.get(key);
                        if (cached != null) {
                            tokenSim = cached.doubleValue();
                        } else {
                            tokenSim = jaroWinklerDistance.scoreStrings(sVal, tVals[j]);
                            tokenSimCache.put(key, tokenSim);
                        }
                    }

                    if (tokenSim >= minTokenSimilarity) {
                        double ws = tokenSim * sw * tWeights[j];
                        if (ws > best) {
                            best = ws;
                            bestI = i;
                            bestJ = j;
                        }
                    }
                }
            }

            if (bestI < 0) break;
            sim += best;
            sc.sUsed[bestI] = true;
            sc.tUsed[bestJ] = true;
            remaining--;
        }

        return sim;
    }

    @Override
    public String getResult(int i) { return result.get(i).found; }
    @Override
    public Object getValue(int i) { return result.get(i).value; }
    @Override
    public double getScore(int i) { return result.get(i).score; }
}
