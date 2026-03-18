package com.wcohen.ss.fast;

import jdk.incubator.vector.*;

/**
 * Vectorized Jaro distance using the Java Vector API (JDK 21 incubator).
 * Uses char arrays and boolean masks instead of string operations.
 */
public final class VectorizedJaro {

    private static final VectorSpecies<Short> SHORT_SPECIES = ShortVector.SPECIES_PREFERRED;

    private static final ThreadLocal<ScratchBuffers> SCRATCH = ThreadLocal.withInitial(ScratchBuffers::new);

    private static final class ScratchBuffers {
        boolean[] s2Matched = new boolean[4096];
        char[] common1 = new char[4096];
    }

    public static double score(String str1, String str2) {
        int len1 = str1.length();
        int len2 = str2.length();
        if (len1 == 0 || len2 == 0) return 0.0;

        int halfLen = (Math.min(len1, len2) / 2) + 1;

        ScratchBuffers scratch = SCRATCH.get();
        if (scratch.s2Matched.length < len2) {
            scratch.s2Matched = new boolean[len2 + 256];
        }
        if (scratch.common1.length < Math.min(len1, len2)) {
            scratch.common1 = new char[Math.min(len1, len2) + 256];
        }
        boolean[] s2Matched = scratch.s2Matched;
        char[] common1 = scratch.common1;

        for (int i = 0; i < len2; i++) s2Matched[i] = false;

        int common1Len = 0;

        for (int i = 0; i < len1; i++) {
            char ch = str1.charAt(i);
            int lo = Math.max(0, i - halfLen);
            int hi = Math.min(i + halfLen + 1, len2);
            for (int j = lo; j < hi; j++) {
                if (!s2Matched[j] && str2.charAt(j) == ch) {
                    s2Matched[j] = true;
                    common1[common1Len++] = ch;
                    break;
                }
            }
        }

        if (common1Len == 0) return 0.0;

        char[] common2 = new char[common1Len];
        int common2Len = 0;
        for (int i = 0; i < len2 && common2Len < common1Len; i++) {
            if (s2Matched[i]) {
                common2[common2Len++] = str2.charAt(i);
            }
        }

        if (common2Len != common1Len) return 0.0;

        int transpositions = countTranspositionsVectorized(common1, common2, common1Len);

        return (common1Len / (double) len1 +
                common1Len / (double) len2 +
                (common1Len - transpositions) / (double) common1Len) / 3.0;
    }

    private static int countTranspositionsVectorized(char[] common1, char[] common2, int len) {
        int transpositions = 0;
        int i = 0;
        int bound = SHORT_SPECIES.loopBound(len);

        for (; i < bound; i += SHORT_SPECIES.length()) {
            ShortVector v1 = ShortVector.fromCharArray(SHORT_SPECIES, common1, i);
            ShortVector v2 = ShortVector.fromCharArray(SHORT_SPECIES, common2, i);
            VectorMask<Short> neq = v1.compare(VectorOperators.NE, v2);
            transpositions += neq.trueCount();
        }

        for (; i < len; i++) {
            if (common1[i] != common2[i]) transpositions++;
        }

        return transpositions / 2;
    }
}
