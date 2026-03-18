package com.wcohen.ss.fast;

import com.wcohen.ss.api.Token;

/**
 * Optimized bag-of-tokens with primitive arrays and O(1) token-weight lookup.
 */
public final class FastBagOfTokens {

    private final String original;
    private final Token[] tokens;
    private final int[] tokenIds;
    private final String[] tokenValues;
    private final double[] weights;
    private final double[] weightByIndex;
    private final int size;

    public FastBagOfTokens(String s, Token[] tokens, int maxTokenIndex) {
        this.original = s;
        this.tokens = tokens;

        this.weightByIndex = new double[maxTokenIndex + 1];
        int distinct = 0;
        for (Token tok : tokens) {
            int idx = tok.getIndex();
            if (weightByIndex[idx] == 0.0) distinct++;
            weightByIndex[idx] += 1.0;
        }

        this.tokenIds = new int[distinct];
        this.tokenValues = new String[distinct];
        this.weights = new double[distinct];

        int pos = 0;
        for (Token tok : tokens) {
            int idx = tok.getIndex();
            if (weightByIndex[idx] > 0.0) {
                tokenIds[pos] = idx;
                tokenValues[pos] = tok.getValue();
                weights[pos] = weightByIndex[idx];
                pos++;
                weightByIndex[idx] = -weightByIndex[idx];
            }
        }
        for (int i = 0; i < pos; i++) {
            weightByIndex[tokenIds[i]] = weights[i];
        }
        this.size = distinct;
    }

    public String unwrap() { return original; }
    public Token[] getTokens() { return tokens; }
    public int size() { return size; }
    public int[] getTokenIds() { return tokenIds; }
    public String[] getTokenValues() { return tokenValues; }
    public double[] getWeights() { return weights; }

    public double getWeight(int tokenIndex) {
        return tokenIndex >= 0 && tokenIndex < weightByIndex.length ? weightByIndex[tokenIndex] : 0.0;
    }

    public void setWeight(int tokenIndex, double w) {
        if (tokenIndex < 0 || tokenIndex >= weightByIndex.length) return;
        weightByIndex[tokenIndex] = w;
        for (int i = 0; i < size; i++) {
            if (tokenIds[i] == tokenIndex) {
                weights[i] = w;
                break;
            }
        }
    }

    public void toTFIDFUnitVector(int collectionSize, int[] documentFrequencies) {
        double norm = 0.0;
        for (int i = 0; i < size; i++) {
            int idx = tokenIds[i];
            int df = idx < documentFrequencies.length ? documentFrequencies[idx] : 0;
            if (df == 0) df = 1;
            double tf = weights[i];
            double w = Math.log(tf + 1.0) * Math.log((double) collectionSize / (double) df);
            weights[i] = w;
            weightByIndex[idx] = w;
            norm += w * w;
        }
        norm = Math.sqrt(norm);
        if (norm > 0.0) {
            for (int i = 0; i < size; i++) {
                weights[i] /= norm;
                weightByIndex[tokenIds[i]] = weights[i];
            }
        }
    }

    public double dotProduct(FastBagOfTokens other) {
        FastBagOfTokens small = this.size <= other.size ? this : other;
        FastBagOfTokens large = this.size <= other.size ? other : this;
        double sim = 0.0;
        for (int i = 0; i < small.size; i++) {
            int id = small.tokenIds[i];
            double w2 = large.getWeight(id);
            if (w2 != 0.0) sim += small.weights[i] * w2;
        }
        return sim;
    }
}
