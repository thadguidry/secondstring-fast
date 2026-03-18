package com.wcohen.ss.fast;

import com.wcohen.ss.AbstractStringDistance;
import com.wcohen.ss.BasicStringWrapper;
import com.wcohen.ss.api.StringWrapper;
import com.wcohen.ss.api.StringWrapperIterator;
import com.wcohen.ss.api.Token;

/**
 * Optimized SoftTFIDF using primitive loops and FastJaroWinkler.
 */
public class FastSoftTFIDF extends AbstractStringDistance {

    private final FastSimpleTokenizer tokenizer;
    private final FastJaroWinkler jaroWinkler;
    private final double tokenMatchThreshold;
    private final FastTFIDF tfidf;

    private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(Scratch::new);

    private static final class Scratch {
        boolean[] sUsed = new boolean[64];
        boolean[] tUsed = new boolean[64];
    }

    public FastSoftTFIDF() {
        this(FastSimpleTokenizer.DEFAULT_TOKENIZER, new FastJaroWinkler(), 0.9);
    }

    public FastSoftTFIDF(FastSimpleTokenizer tokenizer, FastJaroWinkler jaroWinkler, double threshold) {
        this.tokenizer = tokenizer;
        this.jaroWinkler = jaroWinkler;
        this.tokenMatchThreshold = threshold;
        this.tfidf = new FastTFIDF(tokenizer);
    }

    public void train(String[] strings) { tfidf.train(strings); }
    public void train(StringWrapperIterator iter) { tfidf.train(iter); }
    public void setDocumentFrequency(Token tok, int df) { tfidf.setDocumentFrequency(tok, df); }
    public void setCollectionSize(int n) { tfidf.setCollectionSize(n); }

    @Override
    public StringWrapper prepare(String s) {
        tfidf.prepare(s);
        return new BasicStringWrapper(s);
    }

    public FastBagOfTokens prepareVector(String s) { return tfidf.prepareVector(s); }

    @Override
    public double score(StringWrapper s, StringWrapper t) {
        return scoreStrings(s.unwrap(), t.unwrap());
    }

    public double scoreStrings(String s, String t) {
        return scoreBags(tfidf.prepareVector(s), tfidf.prepareVector(t));
    }

    public double scoreBags(FastBagOfTokens sBag, FastBagOfTokens tBag) {
        int sSize = sBag.size();
        int tSize = tBag.size();
        if (sSize == 0 || tSize == 0) return 0.0;

        String[] sVals = sBag.getTokenValues();
        String[] tVals = tBag.getTokenValues();
        double[] sWeights = sBag.getWeights();
        double[] tWeights = tBag.getWeights();

        Scratch scratch = SCRATCH.get();
        if (scratch.sUsed.length < sSize) scratch.sUsed = new boolean[sSize];
        if (scratch.tUsed.length < tSize) scratch.tUsed = new boolean[tSize];
        for (int i = 0; i < sSize; i++) scratch.sUsed[i] = false;
        for (int j = 0; j < tSize; j++) scratch.tUsed[j] = false;

        double sim = 0.0;
        int remaining = Math.min(sSize, tSize);

        while (remaining > 0) {
            double best = -1.0;
            int bestI = -1, bestJ = -1;

            for (int i = 0; i < sSize; i++) {
                if (scratch.sUsed[i]) continue;
                String sVal = sVals[i];
                double sw = sWeights[i];
                for (int j = 0; j < tSize; j++) {
                    if (scratch.tUsed[j]) continue;
                    String tVal = tVals[j];
                    double tokenSim = sVal.equals(tVal) ? 1.0 : jaroWinkler.scoreStrings(sVal, tVal);
                    if (tokenSim >= tokenMatchThreshold) {
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
            scratch.sUsed[bestI] = true;
            scratch.tUsed[bestJ] = true;
            remaining--;
        }

        return sim;
    }

    public FastTFIDF getTFIDF() { return tfidf; }
    public FastSimpleTokenizer getTokenizer() { return tokenizer; }

    @Override
    public String explainScore(StringWrapper s, StringWrapper t) {
        return "FastSoftTFIDF score: " + score(s, t);
    }
}
