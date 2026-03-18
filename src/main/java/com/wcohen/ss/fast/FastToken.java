package com.wcohen.ss.fast;

import com.wcohen.ss.api.Token;

/**
 * Token implementation used by fast tokenizer.
 */
public final class FastToken implements Token, Comparable<FastToken> {
    private final int index;
    private final String value;

    public FastToken(int index, String value) {
        this.index = index;
        this.value = value;
    }

    @Override
    public String getValue() { return value; }

    @Override
    public int getIndex() { return index; }

    @Override
    public int compareTo(FastToken o) { return Integer.compare(index, o.index); }

    @Override
    public int hashCode() { return value.hashCode(); }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Token)) return false;
        return value.equals(((Token) o).getValue());
    }

    @Override
    public String toString() { return "[fastTok " + index + ":" + value + "]"; }
}
