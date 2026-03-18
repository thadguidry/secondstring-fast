package com.wcohen.ss.fast;

import com.wcohen.ss.AbstractStringDistance;
import com.wcohen.ss.BasicStringWrapper;
import com.wcohen.ss.api.StringWrapper;
import com.wcohen.ss.api.StringWrapperIterator;
import com.wcohen.ss.api.Token;

/**
 * Optimized TFIDF using primitive df storage and FastBagOfTokens.
 */
public class FastTFIDF extends AbstractStringDistance {

    private final FastSimpleTokenizer tokenizer;
    private int[] documentFrequency;
    private int collectionSize = 0;
    private FastBagOfTokens lastVector;

    public FastTFIDF() {
        this(FastSimpleTokenizer.DEFAULT_TOKENIZER);
    }

    public FastTFIDF(FastSimpleTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.documentFrequency = new int[4096];
    }

    private void ensureCapacity(int idx) {
        if (idx >= documentFrequency.length) {
            int[] next = new int[Math.max(idx + 1, documentFrequency.length * 2)];
            System.arraycopy(documentFrequency, 0, next, 0, documentFrequency.length);
            documentFrequency = next;
        }
    }

    public void train(String[] strings) {
        boolean[] seen = new boolean[documentFrequency.length];
        for (String s : strings) {
            Token[] tokens = tokenizer.tokenize(s);
            for (Token tok : tokens) {
                int idx = tok.getIndex();
                if (idx >= seen.length) {
                    boolean[] nextSeen = new boolean[Math.max(idx + 1, seen.length * 2)];
                    System.arraycopy(seen, 0, nextSeen, 0, seen.length);
                    seen = nextSeen;
                }
                ensureCapacity(idx);
                if (!seen[idx]) {
                    seen[idx] = true;
                    documentFrequency[idx]++;
                }
            }
            for (Token tok : tokens) seen[tok.getIndex()] = false;
            collectionSize++;
        }
    }

    public void train(StringWrapperIterator iter) {
        while (iter.hasNext()) {
            train(new String[] { iter.nextStringWrapper().unwrap() });
        }
    }

    @Override
    public StringWrapper prepare(String s) {
        lastVector = prepareVector(s);
        return new BasicStringWrapper(s);
    }

    public FastBagOfTokens prepareVector(String s) {
        FastBagOfTokens bag = new FastBagOfTokens(s, tokenizer.tokenize(s), tokenizer.maxTokenIndex());
        if (collectionSize > 0) bag.toTFIDFUnitVector(collectionSize, documentFrequency);
        return bag;
    }

    @Override
    public double score(StringWrapper s, StringWrapper t) {
        return prepareVector(s.unwrap()).dotProduct(prepareVector(t.unwrap()));
    }

    @Override
    public String explainScore(StringWrapper s, StringWrapper t) {
        return "FastTFIDF score: " + score(s, t);
    }

    public Token[] getTokens() { return lastVector == null ? new Token[0] : lastVector.getTokens(); }
    public double getWeight(Token tok) { return lastVector == null ? 0.0 : lastVector.getWeight(tok.getIndex()); }
    public int getDocumentFrequency(Token tok) {
        int i = tok.getIndex();
        return i < documentFrequency.length ? documentFrequency[i] : 0;
    }
    public void setDocumentFrequency(Token tok, int df) {
        ensureCapacity(tok.getIndex());
        documentFrequency[tok.getIndex()] = df;
    }
    public int getCollectionSize() { return collectionSize; }
    public void setCollectionSize(int n) { collectionSize = n; }
    public FastSimpleTokenizer getTokenizer() { return tokenizer; }
}
