using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace HNSW.Net.SiftBenchmark
{
    public static class L2Distance
    {
        private static readonly int _vs1 = Vector<float>.Count;
        private static readonly int _vs2 = 2 * Vector<float>.Count;
        private static readonly int _vs3 = 3 * Vector<float>.Count;
        private static readonly int _vs4 = 4 * Vector<float>.Count;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SIMD(float[] u, float[] v)
        {
            float result = 0f;
            var count = u.Length;
            var offset = 0;

            while (count >= _vs4)
            {
                var v_u1 = new Vector<float>(u, offset);
                var v_v1 = new Vector<float>(v, offset);
                var diff1 = Vector.Subtract(v_u1, v_v1);
                result += Vector.Dot(diff1, diff1);

                var v_u2 = new Vector<float>(u, offset + _vs1);
                var v_v2 = new Vector<float>(v, offset + _vs1);
                var diff2 = Vector.Subtract(v_u2, v_v2);
                result += Vector.Dot(diff2, diff2);

                var v_u3 = new Vector<float>(u, offset + _vs2);
                var v_v3 = new Vector<float>(v, offset + _vs2);
                var diff3 = Vector.Subtract(v_u3, v_v3);
                result += Vector.Dot(diff3, diff3);

                var v_u4 = new Vector<float>(u, offset + _vs3);
                var v_v4 = new Vector<float>(v, offset + _vs3);
                var diff4 = Vector.Subtract(v_u4, v_v4);
                result += Vector.Dot(diff4, diff4);

                if (count == _vs4) return result;
                count -= _vs4;
                offset += _vs4;
            }

            if (count >= _vs2)
            {
                var diff1 = Vector.Subtract(new Vector<float>(u, offset), new Vector<float>(v, offset));
                result += Vector.Dot(diff1, diff1);
                var diff2 = Vector.Subtract(new Vector<float>(u, offset + _vs1), new Vector<float>(v, offset + _vs1));
                result += Vector.Dot(diff2, diff2);
                if (count == _vs2) return result;
                count -= _vs2;
                offset += _vs2;
            }
            if (count >= _vs1)
            {
                var diff1 = Vector.Subtract(new Vector<float>(u, offset), new Vector<float>(v, offset));
                result += Vector.Dot(diff1, diff1);
                if (count == _vs1) return result;
                count -= _vs1;
                offset += _vs1;
            }
            while (count > 0)
            {
                float diff = u[offset] - v[offset];
                result += diff * diff;
                offset++;
                count--;
            }
            return result;
        }

        public static float NonOptimized(float[] u, float[] v)
        {
            float result = 0f;
            for (int i = 0; i < u.Length; i++)
            {
                float diff = u[i] - v[i];
                result += diff * diff;
            }
            return result;
        }
    }
}
