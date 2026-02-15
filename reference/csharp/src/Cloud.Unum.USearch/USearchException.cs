using System;

namespace Cloud.Unum.USearch;

public class USearchException : Exception
{
    public USearchException(string message) : base(message) { }
}
