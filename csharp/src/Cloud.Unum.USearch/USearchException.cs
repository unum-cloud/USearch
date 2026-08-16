using System;

namespace Cloud.Unum.USearch;

/// <summary>
/// One failed call: a <see cref="USearch.Status"/> every caller can branch on, and the detail
/// the status cannot carry. Faults this binding catches before the boundary report the status
/// the engine would have reported for the same fault.
/// </summary>
public class USearchException : Exception
{
    /// <summary>
    /// Initializes a failure carrying a status and its detail.
    /// </summary>
    /// <param name="status">What went wrong.</param>
    /// <param name="message">The detail the status cannot carry.</param>
    public USearchException(Status status, string message) : base($"usearch: {status}: {message}")
    {
        this.Status = status;
    }

    /// <summary>
    /// Initializes a failure carrying a status, its detail, and the exception that caused it.
    /// </summary>
    /// <param name="status">What went wrong.</param>
    /// <param name="message">The detail the status cannot carry.</param>
    /// <param name="cause">The exception this failure wraps.</param>
    public USearchException(Status status, string message, Exception cause)
        : base($"usearch: {status}: {message}", cause)
    {
        this.Status = status;
    }

    /// <summary>
    /// What went wrong.
    /// </summary>
    public Status Status { get; }
}
