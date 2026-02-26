from __future__ import annotations

from copy import deepcopy
from numbers import Real
from typing import Iterable, List, Sequence, Tuple, Union, Optional


Number = Union[int, float]


class Matrix:
    """
    A pure-Python 2D matrix class for core linear algebra operations.

    Design notes:
    - Zero third-party dependencies (no numpy/scipy).
    - Floating-point tolerance (epsilon) is used for near-zero and equality checks.
    - Operations like transpose()/inv() return new Matrix instances (immutability-friendly).
    """

    epsilon: float = 1e-7

    def __init__(self, data: Sequence[Sequence[Real]]):
        """
        Initialize a Matrix from a rectangular 2D sequence.

        Args:
            data: A 2D sequence (e.g., list of lists). Each row must have the same length.

        Raises:
            ValueError: If rows have inconsistent lengths, or data is empty/ill-formed.
            TypeError: If any element cannot be converted to float.
        """
        if not isinstance(data, Sequence) or len(data) == 0:
            raise ValueError("data must be a non-empty 2D sequence")

        rows: List[List[float]] = []
        rowlen: Optional[int] = None

        for r, row in enumerate(data):
            if not isinstance(row, Sequence) or len(row) == 0:
                raise ValueError("each row must be a non-empty sequence")
            if rowlen is None:
                rowlen = len(row)
            elif len(row) != rowlen:
                raise ValueError("all rows must have the same length")

            newrow: List[float] = []
            for c, x in enumerate(row):
                try:
                    newrow.append(float(x))
                except Exception as e:
                    raise TypeError(f"matrix element at ({r},{c}) is not numeric") from e
            rows.append(newrow)

        self._data: List[List[float]] = rows

    # -------------------------
    # Magic methods
    # -------------------------

    def __add__(self, other: "Matrix") -> "Matrix":
        """A + B -> calls add()."""
        return self.add(other)

    def __sub__(self, other: "Matrix") -> "Matrix":
        """A - B -> calls sub()."""
        return self.sub(other)

    def __mul__(self, other: Union["Matrix", Real]) -> "Matrix":
        """
        A * k (scalar) -> calls scale()
        A * B (matrix) -> calls dot()
        """
        if isinstance(other, Matrix):
            return self.dot(other)
        if isinstance(other, Real):
            return self.scale(float(other))
        raise TypeError("unsupported operand type(s) for *: 'Matrix' and '{}'".format(type(other).__name__))

    def __rmul__(self, other: Real) -> "Matrix":
        """k * A -> scalar scaling."""
        if isinstance(other, Real):
            return self.scale(float(other))
        raise TypeError("unsupported operand type(s) for *: '{}' and 'Matrix'".format(type(other).__name__))

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """A @ B -> calls dot()."""
        return self.dot(other)

    def __repr__(self) -> str:
        m, n = self.shape()
        return f"Matrix({m}x{n}, data={self._data})"

    # -------------------------
    # Properties & generation
    # -------------------------

    def shape(self) -> Tuple[int, int]:
        """
        Return (m, n) where m is row count and n is column count.
        """
        return (len(self._data), len(self._data[0]))

    def clone(self) -> "Matrix":
        """
        Return a deep copy of this matrix.
        """
        return Matrix(deepcopy(self._data))

    @classmethod
    def eye(cls, n: int) -> "Matrix":
        """
        Create an n x n identity matrix.

        Args:
            n: Size of the identity matrix.

        Raises:
            ValueError: If n <= 0.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        data = [[0.0] * n for _ in range(n)]
        for i in range(n):
            data[i][i] = 1.0
        return cls(data)

    @classmethod
    def zeros(cls, m: int, n: int) -> "Matrix":
        """
        Create an m x n zero matrix.

        Args:
            m: Row count.
            n: Column count.

        Raises:
            ValueError: If m <= 0 or n <= 0.
        """
        if m <= 0 or n <= 0:
            raise ValueError("m and n must be positive")
        return cls([[0.0] * n for _ in range(m)])

    # -------------------------
    # Basic operations
    # -------------------------

    def add(self, other: "Matrix") -> "Matrix":
        """
        Element-wise matrix addition.

        Args:
            other: Another matrix with the same shape.

        Raises:
            ValueError: If shapes do not match.
        """
        self._checkshape(other)
        m, n = self.shape()
        out = [[self._data[i][j] + other._data[i][j] for j in range(n)] for i in range(m)]
        return Matrix(out)

    def sub(self, other: "Matrix") -> "Matrix":
        """
        Element-wise matrix subtraction.

        Args:
            other: Another matrix with the same shape.

        Raises:
            ValueError: If shapes do not match.
        """
        self._checkshape(other)
        m, n = self.shape()
        out = [[self._data[i][j] - other._data[i][j] for j in range(n)] for i in range(m)]
        return Matrix(out)

    def scale(self, k: Real) -> "Matrix":
        """
        Scalar multiplication: k * A.

        Args:
            k: Scalar multiplier.
        """
        kk = float(k)
        m, n = self.shape()
        out = [[self._data[i][j] * kk for j in range(n)] for i in range(m)]
        return Matrix(out)

    def dot(self, other: "Matrix") -> "Matrix":
        """
        Matrix multiplication: A x B.

        Args:
            other: Right-hand matrix.

        Raises:
            ValueError: If inner dimensions do not match.
        """
        m, n = self.shape()
        p, q = other.shape()
        if n != p:
            raise ValueError(f"dimension mismatch: {m}x{n} cannot dot {p}x{q}")

        out = [[0.0] * q for _ in range(m)]
        # Classic triple loop (with a small optimization by reusing row/col access)
        for i in range(m):
            row = self._data[i]
            for k in range(n):
                aik = row[k]
                if abs(aik) <= self.epsilon:
                    continue
                ok = other._data[k]
                for j in range(q):
                    out[i][j] += aik * ok[j]
        return Matrix(out)

    # -------------------------
    # Transformations & features
    # -------------------------

    def transpose(self) -> "Matrix":
        """
        Return the transpose matrix A^T.
        """
        m, n = self.shape()
        out = [[self._data[i][j] for i in range(m)] for j in range(n)]
        return Matrix(out)

    def trace(self) -> float:
        """
        Return the trace of a square matrix (sum of diagonal elements).

        Raises:
            ValueError: If the matrix is not square.
        """
        self._checksquare()
        n = self.shape()[0]
        return sum(self._data[i][i] for i in range(n))

    def rank(self) -> int:
        """
        Compute the rank of the matrix using Gaussian elimination to row-echelon form,
        counting pivots with epsilon tolerance.

        Returns:
            The rank as an integer.
        """
        a = self._copy()
        m, n = self.shape()
        r = 0
        c = 0

        while r < m and c < n:
            # Find pivot row (partial pivoting for stability)
            piv = self._pivot(a, r, c)
            if piv is None:
                c += 1
                continue
            if piv != r:
                a[r], a[piv] = a[piv], a[r]

            pivot = a[r][c]
            # Eliminate below
            for i in range(r + 1, m):
                if abs(a[i][c]) <= self.epsilon:
                    continue
                factor = a[i][c] / pivot
                a[i][c] = 0.0
                for j in range(c + 1, n):
                    a[i][j] -= factor * a[r][j]

            r += 1
            c += 1

        # Count non-zero rows (in echelon form, pivots correspond to rank)
        rank = 0
        for i in range(m):
            if any(abs(a[i][j]) > self.epsilon for j in range(n)):
                rank += 1
        return rank

    # -------------------------
    # Core linear algebra
    # -------------------------

    def det(self) -> float:
        """
        Compute determinant det(A) for a square matrix using Gaussian elimination
        with partial pivoting (more stable than Laplace expansion).

        Algorithm idea:
        - Convert A to upper-triangular U via elimination.
        - det(A) = sign * product(diagonal(U)), where sign flips on each row swap.

        Returns:
            Determinant as a float.

        Raises:
            ValueError: If the matrix is not square.
        """
        self._checksquare()
        a = self._copy()
        n = self.shape()[0]
        sign = 1.0

        for c in range(n):
            piv = self._pivot(a, c, c)
            if piv is None:
                return 0.0
            if piv != c:
                a[c], a[piv] = a[piv], a[c]
                sign *= -1.0

            pivot = a[c][c]
            if abs(pivot) <= self.epsilon:
                return 0.0

            # Eliminate below pivot
            for r in range(c + 1, n):
                if abs(a[r][c]) <= self.epsilon:
                    continue
                factor = a[r][c] / pivot
                a[r][c] = 0.0
                for j in range(c + 1, n):
                    a[r][j] -= factor * a[c][j]

        detv = sign
        for i in range(n):
            detv *= a[i][i]
        # Treat very small results as zero (tolerance)
        return 0.0 if abs(detv) <= self.epsilon else detv

    def minor(self, i: int, j: int) -> float:
        """
        Compute the minor determinant after removing row i and column j.

        Indices are 0-based.

        Args:
            i: Row index to remove.
            j: Column index to remove.

        Returns:
            det(M_ij) as float, where M_ij is the submatrix.

        Raises:
            ValueError: If matrix is not square or indices out of range.
        """
        self._checksquare()
        n = self.shape()[0]
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError("minor indices out of range")
        sub = []
        for r in range(n):
            if r == i:
                continue
            row = []
            for c in range(n):
                if c == j:
                    continue
                row.append(self._data[r][c])
            sub.append(row)
        return Matrix(sub).det()

    def cofactor(self, i: int, j: int) -> float:
        """
        Compute the cofactor C_ij = (-1)^(i+j) * minor(i, j).

        Indices are 0-based.
        """
        s = -1.0 if (i + j) % 2 == 1 else 1.0
        return s * self.minor(i, j)

    def adjoint(self) -> "Matrix":
        """
        Return the adjoint (adjugate) matrix: adj(A) = C^T,
        where C is the cofactor matrix.

        Raises:
            ValueError: If matrix is not square.
        """
        self._checksquare()
        n = self.shape()[0]
        cof = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                cof[i][j] = self.cofactor(i, j)
        return Matrix(cof).transpose()

    def inv(self) -> "Matrix":
        """
        Compute the inverse A^{-1} using Gauss-Jordan elimination with partial pivoting.

        Algorithm idea:
        - Form augmented matrix [A | I]
        - Row-reduce left side to I, applying the same operations to the right side.
        - Right side becomes A^{-1}.

        Returns:
            The inverse matrix.

        Raises:
            ValueError: If matrix is not square or is singular (non-invertible).
        """
        self._checksquare()
        n = self.shape()[0]

        # Build augmented matrix [A | I]
        aug = [self._data[i][:] + Matrix.eye(n)._data[i][:] for i in range(n)]

        for c in range(n):
            # Pivot selection (max abs value in column c)
            piv = self._pivot(aug, c, c)
            if piv is None:
                raise ValueError("matrix is singular (pivot not found)")
            if piv != c:
                aug[c], aug[piv] = aug[piv], aug[c]

            pivot = aug[c][c]
            if abs(pivot) <= self.epsilon:
                raise ValueError("matrix is singular (zero pivot)")

            # Normalize pivot row
            invp = 1.0 / pivot
            for j in range(2 * n):
                aug[c][j] *= invp

            # Eliminate other rows
            for r in range(n):
                if r == c:
                    continue
                factor = aug[r][c]
                if abs(factor) <= self.epsilon:
                    aug[r][c] = 0.0
                    continue
                aug[r][c] = 0.0
                for j in range(c + 1, 2 * n):
                    aug[r][j] -= factor * aug[c][j]

        invdata = [row[n:] for row in aug]
        return Matrix(invdata)

    # -------------------------
    # Advanced
    # -------------------------

    def solve(self, b: Union[Sequence[Real], "Matrix"]) -> "Matrix":
        """
        Solve the linear system Ax = b for x using Gaussian elimination with partial pivoting.

        Args:
            b:
              - A length-n sequence (treated as an n x 1 column vector), or
              - A Matrix with shape (n, k) for multiple right-hand sides.

        Returns:
            Solution matrix x with shape (n, 1) or (n, k).

        Raises:
            ValueError: If A is not square, dimensions mismatch, or the system is singular.
        """
        self._checksquare()
        n = self.shape()[0]

        if isinstance(b, Matrix):
            br, bc = b.shape()
            if br != n:
                raise ValueError("dimension mismatch: b must have the same row count as A")
            rhs = b._copy()
            k = bc
        else:
            if len(b) != n:
                raise ValueError("dimension mismatch: b length must match A size")
            rhs = [[float(x)] for x in b]
            k = 1

        a = self._copy()

        # Forward elimination to upper-triangular form
        for c in range(n):
            piv = self._pivot(a, c, c)
            if piv is None:
                raise ValueError("system is singular (pivot not found)")
            if piv != c:
                a[c], a[piv] = a[piv], a[c]
                rhs[c], rhs[piv] = rhs[piv], rhs[c]

            pivot = a[c][c]
            if abs(pivot) <= self.epsilon:
                raise ValueError("system is singular (zero pivot)")

            for r in range(c + 1, n):
                if abs(a[r][c]) <= self.epsilon:
                    a[r][c] = 0.0
                    continue
                factor = a[r][c] / pivot
                a[r][c] = 0.0
                for j in range(c + 1, n):
                    a[r][j] -= factor * a[c][j]
                for t in range(k):
                    rhs[r][t] -= factor * rhs[c][t]

        # Back substitution
        x = [[0.0] * k for _ in range(n)]
        for i in range(n - 1, -1, -1):
            pivot = a[i][i]
            if abs(pivot) <= self.epsilon:
                raise ValueError("system is singular (zero pivot during back substitution)")
            for t in range(k):
                s = rhs[i][t]
                for j in range(i + 1, n):
                    s -= a[i][j] * x[j][t]
                x[i][t] = s / pivot

        return Matrix(x)

    def decompose(self) -> Tuple["Matrix", "Matrix"]:
        """
        Compute LU decomposition (Doolittle) WITHOUT pivoting: A = L U.

        Returns:
            (L, U) where:
              - L is lower triangular with 1s on the diagonal,
              - U is upper triangular.

        Raises:
            ValueError: If matrix is not square, or a zero/near-zero pivot is encountered.
        """
        self._checksquare()
        n = self.shape()[0]
        a = self._copy()

        L = Matrix.eye(n)._data
        U = Matrix.zeros(n, n)._data

        for i in range(n):
            # Compute U[i][j] for j >= i
            for j in range(i, n):
                s = 0.0
                for k in range(i):
                    s += L[i][k] * U[k][j]
                U[i][j] = a[i][j] - s

            pivot = U[i][i]
            if abs(pivot) <= self.epsilon:
                raise ValueError("LU decomposition failed (zero pivot); try a different method with pivoting")

            # Compute L[j][i] for j > i
            for j in range(i + 1, n):
                s = 0.0
                for k in range(i):
                    s += L[j][k] * U[k][i]
                L[j][i] = (a[j][i] - s) / pivot

        return Matrix(L), Matrix(U)

    # -------------------------
    # Internal helpers (private)
    # -------------------------

    def _checkshape(self, other: "Matrix") -> None:
        if not isinstance(other, Matrix):
            raise TypeError("other must be a Matrix")
        if self.shape() != other.shape():
            raise ValueError("shape mismatch")

    def _checksquare(self) -> None:
        m, n = self.shape()
        if m != n:
            raise ValueError("matrix must be square")

    def _copy(self) -> List[List[float]]:
        return [row[:] for row in self._data]

    def _pivot(self, a: List[List[float]], start: int, col: int) -> Optional[int]:
        """
        Return the row index of the best pivot in column 'col' at or below 'start',
        using max-absolute-value selection. Return None if all candidates are ~0.
        """
        best = None
        bestv = 0.0
        for r in range(start, len(a)):
            v = abs(a[r][col])
            if v > bestv:
                bestv = v
                best = r
        if best is None or bestv <= self.epsilon:
            return None
        return best

    def power_iteration(self, num_simulations: int = 100) -> Tuple[float, "Matrix"]:
        """
        使用幂法 (Power Iteration) 求方阵的主特征值 (绝对值最大的特征值) 及其对应的特征向量。
        这在工程实践 (如 PCA 获取第一主成分、搜索引擎的 PageRank 算法) 中是极其核心的底层算法。

        Args:
            num_simulations: 迭代次数。通常 100 次对于常规矩阵已足够收敛。

        Returns:
            (eigenvalue, eigenvector): 一个元组，包含特征值 (float) 和 对应的特征向量 (n x 1 的 Matrix 对象)。

        Raises:
            ValueError: 如果矩阵不是方阵。
        """
        self._checksquare()
        n = self.shape()[0]

        # 1. 初始猜测向量：为了兼容现有的库，我们生成一个 n x 1 的全 1 列向量矩阵
        b_k = Matrix([[1.0] for _ in range(n)])

        # 2. 核心迭代过程
        for _ in range(num_simulations):
            # 计算 A * b_k
            b_k1 = self.dot(b_k)

            # 计算 b_k1 的 2-范数 (即向量的欧几里得长度)
            norm_sq = 0.0
            for i in range(n):
                norm_sq += b_k1._data[i][0] ** 2
            norm = norm_sq ** 0.5

            # 防止除以零 (极度奇异或全零矩阵)
            if norm <= self.epsilon:
                break

            # 归一化：将向量长度缩放为 1
            b_k1 = b_k1.scale(1.0 / norm)
            b_k = b_k1

        # 3. 计算瑞利商 (Rayleigh quotient) 提取特征值
        # 特征值 = (b_k^T * A * b_k) / (b_k^T * b_k)
        # 因为 b_k 已经是长度为 1 的单位向量，分母为 1，直接计算分子即可
        Ab_k = self.dot(b_k)
        eigenvalue = b_k.transpose().dot(Ab_k)._data[0][0]

        return eigenvalue, b_k