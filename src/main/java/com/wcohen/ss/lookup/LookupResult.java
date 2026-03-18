package com.wcohen.ss.lookup;

import java.io.*;
import java.util.*;

import com.wcohen.ss.*;
import com.wcohen.ss.api.*;
import com.wcohen.ss.tokens.*;

/**
 * Shared code for SoftTFIDFDictionary and the rescoring variant of it.
 * Made public (was package-private) to allow access from com.wcohen.ss.fast.
 */
public class LookupResult implements Comparable
{
    private static final java.text.DecimalFormat fmt = new java.text.DecimalFormat("0.000");

    public String found; // a string 'looked up' in a dictionary
    public Object value; // the value associated with that string
    public double score; // the score of the match between the looked-up string and 'found'

    public LookupResult(String found,Object value,double score)
    {
        this.found=found; this.value=value; this.score=score;
    }

    public int compareTo(Object o)
    {
        double diff = ((LookupResult)o).score - score;
        return diff<0 ? -1 : (diff>0?+1:0);
    }

    public String toString() { return "["+fmt.format(score)+" "+found+"=>"+value+"]"; }
}
