package com.wcohen.ss.fast;

import com.wcohen.ss.Levenstein;
import org.apache.commons.text.similarity.LevenshteinDistance;
import org.junit.jupiter.api.Test;

import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness tests for {@link FastLevenstein} against the reference {@link Levenstein}.
 * Uses the same negative-score convention: score(s,t) == -editDistance(s,t).
 */
class FastLevensteinTest {

    private final Levenstein orig = new Levenstein();
    private final FastLevenstein fast = new FastLevenstein();
    private final LevenshteinDistance commons = LevenshteinDistance.getDefaultInstance();

    // ---- editDistance API ----

    @Test
    void identical() {
        assertEquals(0, FastLevenstein.editDistance("hello", "hello"));
    }

    @Test
    void emptyLeft() {
        assertEquals(5, FastLevenstein.editDistance("", "hello"));
    }

    @Test
    void emptyRight() {
        assertEquals(5, FastLevenstein.editDistance("hello", ""));
    }

    @Test
    void bothEmpty() {
        assertEquals(0, FastLevenstein.editDistance("", ""));
    }

    @Test
    void singleSubstitution() {
        assertEquals(1, FastLevenstein.editDistance("cat", "bat"));
    }

    @Test
    void singleInsertion() {
        assertEquals(1, FastLevenstein.editDistance("cat", "cats"));
    }

    @Test
    void singleDeletion() {
        assertEquals(1, FastLevenstein.editDistance("cats", "cat"));
    }

    @Test
    void completelyDifferent() {
        assertEquals(3, FastLevenstein.editDistance("abc", "xyz"));
    }

    @Test
    void caseInsensitive() {
        // Original Levenstein uses case-insensitive matching via CharMatchScore.DIST_01
        assertEquals(0, FastLevenstein.editDistance("Hello", "hello"));
        assertEquals(0, FastLevenstein.editDistance("WORLD", "world"));
    }

    @Test
    void longerStrings() {
        assertEquals(
            FastLevenstein.editDistance("kitten", "sitting"),
            3  // classic Levenshtein example (case-insensitive: same)
        );
    }

    // ---- score API matches original (negative convention) ----

    @Test
    void scoreMatchesOriginalOnVariousPairs() {
        String[][] pairs = {
            {"", ""},
            {"", "hello"},
            {"hello", ""},
            {"hello", "hello"},
            {"abc", "abc"},
            {"abc", "abcd"},
            {"abc", "xyz"},
            {"kitten", "sitting"},
            {"Sunday", "Saturday"},
            {"California", "Californication"},
            {"OpenRefine", "OpenRefinement"},
            {"the quick brown fox", "the slow green fox"},
        };

        for (String[] p : pairs) {
            double expected = orig.score(p[0], p[1]);
            double actual   = fast.scoreStrings(p[0], p[1]);
            assertEquals(expected, actual, 0.0,
                "Mismatch for (\"" + p[0] + "\", \"" + p[1] + "\")");
        }
    }

    @Test
    void scoreViaStringWrapperAPI() {
        assertEquals(
            orig.score("restaurant", "restraunt"),
            fast.score("restaurant", "restraunt"),
            0.0
        );
    }

    @Test
    void normalizedCommonsDistanceMatchesOriginalNegativeScore() {
        String[][] pairs = {
            {"", ""},
            {"Hello", "hello"},
            {"WORLD", "world"},
            {"kitten", "sitting"},
            {"Sunday", "Saturday"},
            {"California", "Californication"},
            {"OpenRefine", "OpenRefinement"},
            {"the quick brown fox", "the slow green fox"},
        };

        for (String[] p : pairs) {
            int commonsDistance = commons.apply(
                p[0].toLowerCase(Locale.ROOT),
                p[1].toLowerCase(Locale.ROOT)
            );
            assertEquals(
                -commonsDistance,
                orig.score(p[0], p[1]),
                0.0,
                "Mismatch for normalized Commons pair (\"" + p[0] + "\", \"" + p[1] + "\")"
            );
        }
    }

    @Test
    void threadLocalBufferReuseDoesNotCorrupt() {
        // Run many pairs sequentially to stress ThreadLocal reuse
        Levenstein ref = new Levenstein();
        FastLevenstein candidate = new FastLevenstein();
        String[] words = {
            "alpha", "beta", "gamma", "delta", "epsilon",
            "zeta", "eta", "theta", "iota", "kappa"
        };
        for (String a : words) {
            for (String b : words) {
                assertEquals(
                    ref.score(a, b),
                    candidate.scoreStrings(a, b),
                    0.0,
                    "Mismatch for (\"" + a + "\", \"" + b + "\")"
                );
            }
        }
    }

    @Test
    void longStringsDoNotOverflowBuffer() {
        // Strings longer than the initial ThreadLocal buffer size (256)
        String long1 = "a".repeat(400);
        String long2 = "b".repeat(400);
        assertEquals(400, FastLevenstein.editDistance(long1, long2));

        String long3 = "a".repeat(400);
        String long4 = "a".repeat(399) + "b";
        assertEquals(1, FastLevenstein.editDistance(long3, long4));
    }
}
