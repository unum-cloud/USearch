package cloud.unum.usearch;

import java.util.Arrays;

/**
 * A 128-bit entry identity, most significant byte first, rendered as a lowercase dashed UUID.
 *
 * <p>
 * Keys need not be unique: several entries may share one, and {@link DenseIndex#count(Key)}
 * reports how many. Integer identities park in the low eight bytes, so {@code Key.of(42)} prints
 * as {@code 00000000-0000-0000-0000-00000000002a} - the same rendering every other USearch SDK
 * produces for the same sixteen bytes.
 */
public final class Key implements Comparable<Key> {

    /** The width of a key in bytes, and the row pitch of every key matrix crossing the boundary. */
    public static final int BYTES = 16;

    /** The all-zero key. */
    public static final Key NIL = new Key(new byte[BYTES]);

    private static final char[] HEXADECIMAL = "0123456789abcdef".toCharArray();

    private final byte[] bytes;

    private Key(byte[] adopted) {
        this.bytes = adopted;
    }

    /** Parks an integer identity in the low eight bytes, zeroing the rest. */
    public static Key of(long value) {
        byte[] bytes = new byte[BYTES];
        for (int position = 0; position < 8; ++position) {
            bytes[15 - position] = (byte) (value >>> (8 * position));
        }
        return new Key(bytes);
    }

    /** Copies sixteen bytes, most significant first. */
    public static Key ofBytes(byte[] source) {
        return ofBytes(source, 0);
    }

    /** Copies sixteen bytes starting at an offset, most significant first. */
    public static Key ofBytes(byte[] source, int offset) {
        if (source == null) {
            throw new IndexException(Status.MALFORMED_INPUT, "the key borrows no storage");
        }
        if (offset < 0 || source.length - BYTES < offset) {
            throw new IndexException(Status.MALFORMED_INPUT, "the key does not fit at that offset");
        }
        return new Key(Arrays.copyOfRange(source, offset, offset + BYTES));
    }

    /** Reads the rendering {@link #toString()} produces. Dashes may sit anywhere, or nowhere. */
    public static Key parse(String text) {
        if (text == null) {
            throw new IndexException(Status.MALFORMED_INPUT, "the key is not thirty-two hexadecimal digits");
        }
        byte[] digits = new byte[BYTES * 2];
        int written = 0;
        for (int position = 0; position < text.length(); ++position) {
            char character = text.charAt(position);
            if (character == '-') {
                continue;
            }
            if (written == digits.length) {
                throw new IndexException(Status.MALFORMED_INPUT, "the key is not thirty-two hexadecimal digits");
            }
            digits[written++] = hexadecimalDigit(character);
        }
        if (written != digits.length) {
            throw new IndexException(Status.MALFORMED_INPUT, "the key is not thirty-two hexadecimal digits");
        }
        byte[] bytes = new byte[BYTES];
        for (int position = 0; position < BYTES; ++position) {
            bytes[position] = (byte) ((digits[position * 2] << 4) | digits[position * 2 + 1]);
        }
        return new Key(bytes);
    }

    private static byte hexadecimalDigit(char character) {
        if (character >= '0' && character <= '9') {
            return (byte) (character - '0');
        }
        if (character >= 'a' && character <= 'f') {
            return (byte) (character - 'a' + 10);
        }
        if (character >= 'A' && character <= 'F') {
            return (byte) (character - 'A' + 10);
        }
        throw new IndexException(Status.MALFORMED_INPUT, "the key is not thirty-two hexadecimal digits");
    }

    /** Copies the sixteen bytes out, most significant first. */
    public byte[] toBytes() {
        return bytes.clone();
    }

    /** Writes the sixteen bytes into a slab at an offset, most significant first. */
    public void writeInto(byte[] destination, int offset) {
        System.arraycopy(bytes, 0, destination, offset, BYTES);
    }

    /** Reports whether the high eight bytes are all zero, so the key names an integer identity. */
    public boolean isIntegral() {
        for (int position = 0; position < 8; ++position) {
            if (bytes[position] != 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Reports the integer identity parked in the low eight bytes.
     *
     * @throws IndexException when the high eight bytes carry anything, so no integer describes the
     *     key; {@link #isIntegral()} asks first.
     */
    public long toLong() {
        if (!isIntegral()) {
            throw new IndexException(Status.MALFORMED_INPUT, "the key carries more than an integer identity");
        }
        long value = 0;
        for (int position = 8; position < BYTES; ++position) {
            value = (value << 8) | (bytes[position] & 0xffL);
        }
        return value;
    }

    /** Renders the key as a lowercase dashed UUID. */
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder(36);
        for (int position = 0; position < BYTES; ++position) {
            if (position == 4 || position == 6 || position == 8 || position == 10) {
                builder.append('-');
            }
            int digit = bytes[position] & 0xff;
            builder.append(HEXADECIMAL[digit >>> 4]).append(HEXADECIMAL[digit & 0x0f]);
        }
        return builder.toString();
    }

    /** Two keys are equal when their sixteen bytes are. */
    @Override
    public boolean equals(Object other) {
        return other instanceof Key && Arrays.equals(bytes, ((Key) other).bytes);
    }

    /** Hashes the same sixteen bytes {@link #equals(Object)} compares, so the two agree. */
    @Override
    public int hashCode() {
        return Arrays.hashCode(bytes);
    }

    /** Orders keys by their bytes, most significant first - the order their rendering sorts in. */
    @Override
    public int compareTo(Key other) {
        return Arrays.compareUnsigned(bytes, other.bytes);
    }
}
