package com.wcohen.ss.fast;

import com.wcohen.ss.*;
import com.wcohen.ss.api.*;

public class FastJaroWinkler extends AbstractStringDistance {

    private final FastJaro jaro = new FastJaro();

    @Override
    public double score(StringWrapper s, StringWrapper t) {
        double dist = jaro.score(s, t);
        int prefLength = commonPrefixLength(s.unwrap(), t.unwrap());
        return dist + prefLength * 0.1 * (1.0 - dist);
    }

    public double scoreStrings(String s, String t) {
        double dist = FastJaro.scoreStrings(s.toLowerCase(), t.toLowerCase());
        int prefLength = commonPrefixLength(s.toLowerCase(), t.toLowerCase());
        return dist + prefLength * 0.1 * (1.0 - dist);
    }

    private static int commonPrefixLength(String a, String b) {
        int n = Math.min(4, Math.min(a.length(), b.length()));
        for (int i = 0; i < n; i++) {
            if (a.charAt(i) != b.charAt(i)) return i;
        }
        return n;
    }

    @Override
    public String explainScore(StringWrapper s, StringWrapper t) {
        return "FastJaroWinkler score: " + score(s, t);
    }

    @Override
    public StringWrapper prepare(String s) {
        return new BasicStringWrapper(s.toLowerCase());
    }
}
