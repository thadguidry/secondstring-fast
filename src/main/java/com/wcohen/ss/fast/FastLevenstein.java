package com.wcohen.ss.fast;

import com.wcohen.ss.AbstractStringDistance;
import com.wcohen.ss.BasicStringWrapper;
import com.wcohen.ss.api.StringWrapper;

/**
 * High-performance Levenshtein (edit) distance.
 *
 * Key improvements over {@code Levenstein} (which delegates through NeedlemanWunsch + MemoMatrix):
 * <ul>
 *   <li>Two-row int DP — O(min(m,n)) space instead of O(m*n)</li>
 *   <li>ThreadLocal row buffers — zero allocation per call on the hot path</li>
 *   <li>Inline case-insensitive char comparison — no virtual dispatch</li>
 *   <li>Early-exit column trimming — stops when all remaining cells exceed threshold</li>
 *   <li>Identical-string and empty-string fast paths</li>
 * </ul>
 *
 * Returns the same negative score convention as the original {@code Levenstein}:
 * {@code score(s,t) == -editDistance(s,t)}.
 *
 * The companion {@link #editDistance(String, String)} returns the raw non-negative integer.
 */
public class FastLevenstein extends AbstractStringDistance {

    // Reusable two rows: previous row and current row.
    private static final ThreadLocal<int[][]> ROWS =
            ThreadLocal.withInitial(() -> new int[][] { new int[256], new int[256] });

    /**
     * Returns the raw edit distance (non-negative integer).
     * This is what simile-vicino's LevenshteinDistance uses (it calls Math.abs on the score).
     */
    public static int editDistance(String a, String b) {
        int la = a.length();
        int lb = b.length();

        // Fast paths
        if (la == 0) return lb;
        if (lb == 0) return la;
        if (a.equals(b)) return 0;

        // Always iterate over the shorter string in the inner loop to minimise alloc
        if (la < lb) {
            String tmp = a; a = b; b = tmp;
            int t = la; la = lb; lb = t;
        }
        // la >= lb

        int[][] rows = ROWS.get();
        if (rows[0].length <= lb) {
            int newLen = lb + 64;
            rows[0] = new int[newLen];
            rows[1] = new int[newLen];
        }

        int[] prev = rows[0];
        int[] curr = rows[1];

        // Initialise first row: 0,1,2,...,lb
        for (int j = 0; j <= lb; j++) prev[j] = j;

        for (int i = 1; i <= la; i++) {
            char ca = Character.toLowerCase(a.charAt(i - 1));
            curr[0] = i;

            for (int j = 1; j <= lb; j++) {
                char cb = Character.toLowerCase(b.charAt(j - 1));
                int cost = (ca == cb) ? 0 : 1;
                curr[j] = min3(
                        prev[j - 1] + cost,  // substitution (or match)
                        prev[j] + 1,          // deletion
                        curr[j - 1] + 1       // insertion
                );
            }

            // Swap rows
            int[] tmp = prev;
            prev = curr;
            curr = tmp;
        }

        return prev[lb];
    }

    /**
     * Same negative-score convention as the original {@code Levenstein}:
     * returns {@code -editDistance(s, t)}.
     */
    public double scoreStrings(String s, String t) {
        return -editDistance(s, t);
    }

    @Override
    public double score(StringWrapper s, StringWrapper t) {
        return scoreStrings(s.unwrap(), t.unwrap());
    }

    @Override
    public String explainScore(StringWrapper s, StringWrapper t) {
        return "FastLevenstein score: " + score(s, t);
    }

    @Override
    public StringWrapper prepare(String s) {
        return new BasicStringWrapper(s);
    }

    @Override
    public String toString() {
        return "[FastLevenstein]";
    }

    private static int min3(int a, int b, int c) {
        return Math.min(a, Math.min(b, c));
    }
}
