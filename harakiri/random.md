See https://quantumspy.wordpress.com/2014/10/24/generating-random-density-matrices/

ALGORITHM 1
  1. Create a random pure state |\psi\rangle in a higher dimensional Hilbert space \mathcal{H}_{n+m}.
  2. Your random density matrix will be \rho = tr_m[|\psi \rangle \langle \psi |] , where tr_m[\cdot] refers to the partial trace over \mathcal{H}_m.

ALGORITHM 2
  1. Generate some distribution of eigenvalues \{ \lambda_i \} of \rho, with \sum \lambda_i = 1.
  2. Rotate the distribution with a random unitary.

ALGORITHM 3
  1. Create a random Complex matrix H.
  2. Make it positive, and normalize to unity. i.e. \rho = HH^\dagger/(tr[H H^\dagger])

ALGORITHM 4
  First though, you need to fix a basis that spans the whole set,X. For density matrices, a good basis is the one for traceless hermitian matrices:

  Z_k = \delta_{k,k} - \delta_{k+1,k+1} for 1\leq k \leq n-1
  X_{jk} = \delta_{j,k} + \delta_{k,j} for 1\leq j < k \leq n
  Y_{jk} = \mathbf{i} (\delta_{j,k} - \delta_{k,j}) for 1\leq j < k \leq n

  But any normalized basis will do, which for the algorithm I shall call e_i, where the index i\in\{1,N^2-1\}.

  1. Pick a \rho_0 inside your set (i.e. not a pure point!). So, to make life easier, choose it as the maximally mixed state.

  2. Draw a random index i with uniform distribution over \{1,N^2-1\}.

  3. Find the boundary points along this direction. That is, find \xi_1,\xi_2 >0 such that \rho_0 - \xi_1e_i, and \rho_0 + \xi_2e_i lie on the boundary \partial X. In our case, this will be where \rho_0 - \xi_1e_i, and \rho_0 + \xi_2e_i fail to be positive matrices, so it can be done by solving the equation det(\rho_0 - \xi) = 0.

  4. Sample a random number \eta uniformly from the interval (\xi_1,\xi_2).

  5. Your new matrix is \rho_1 = \rho_0 + \eta e_i.

  6. Repeat steps (2)-(5) ad libitum.

  https://arxiv.org/pdf/1312.7061.pdf
