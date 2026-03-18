package com.wcohen.ss.fast;

import com.wcohen.ss.*;
import com.wcohen.ss.api.*;

public class FastJaro extends AbstractStringDistance {

    private static final ThreadLocal<boolean[]> MATCHED_SCRATCH =
            ThreadLocal.withInitial(() -> new boolean[4096]);

    @Override
    public double score(StringWrapper s, StringWrapper t) {
        return scoreStrings(s.unwrap(), t.unwrap());
    }

    public static double scoreStrings(String str1, String str2) {
        int len1 = str1.length();
        int len2 = str2.length();
        if (len1 == 0 || len2 == 0) return 0.0;

        int halfLen = (Math.min(len1, len2) / 2) + 1;

        boolean[] s2Matched = MATCHED_SCRATCH.get();
        if (s2Matched.length < len2) {
            s2Matched = new boolean[len2 + 256];
            MATCHED_SCRATCH.set(s2Matched);
        }
        for (int i = 0; i < len2; i++) s2Matched[i] = false;

        int common1Len = 0;
        int transpositions = 0;
        char[] common1 = new char[Math.min(len1, len2)];

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

        int common2Idx = 0;
        for (int i = 0; i < len2 && common2Idx < common1Len; i++) {
            if (s2Matched[i]) {
                if (str2.charAt(i) != common1[common2Idx]) {
                    transpositions++;
                }
                common2Idx++;
            }
        }

        if (common2Idx != common1Len) return 0.0;
        transpositions /= 2;

        return (common1Len / (double) len1 +
                common1Len / (double) len2 +
                (common1Len - transpositions) / (double) common1Len) / 3.0;
    }

    @Override
    public String explainScore(StringWrapper s, StringWrapper t) {
        return "FastJaro score: " + score(s, t);
    }

    @Override
    public StringWrapper prepare(String s) {
        return new BasicStringWrapper(s.toLowerCase());
    }
}
