function dno_series_gen_data

% Generates training data pairs (eta, xi) -> G(eta)xi for the DNO operator
% Uses single trig mode with analytical solution for error-based truncation

clc; clear all;

% numerical parameters
%%%%%%%%%%%%%%%%%%%%%%

Nx = 1024;
L = 164;

dx = L/Nx;
dk = 2*pi/L;
x = dx*(0:Nx-1)';
k = dk*[0:Nx/2,1-Nx/2:-1]';

% physical parameters
%%%%%%%%%%%%%%%%%%%%%

g = 1;
h = 1;

% data generation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nsamples = 10000;          % number of training samples
a0_min = 0.001;           % min amplitude
a0_max = 0.3;           % max amplitude (keep small for convergence)
steepness_min = 5e-3;    % minimum k0*a0
steepness_max = 0.15;    % maximum k0*a0
n0_max = 20;             % max wavenumber index (k0_max ~ 0.77 on L=164)
n0_min = ceil(steepness_min / (a0_max * dk));

M_max = 8;              % maximum Taylor series order

G0 = k.*tanh(h*k);

% storage for training data
eta_data = zeros(Nx, Nsamples);
xi_data = zeros(Nx, Nsamples);
Gxi_data = zeros(Nx, Nsamples);
M_used = zeros(Nsamples, 1);    % actual truncation order used
final_err = zeros(Nsamples, 1); % final relative error

rng(42);  % for reproducibility

for s = 1:Nsamples
    fprintf('Generating sample %d/%d\n', s, Nsamples);
    
    % random wavenumber index
    n0 = randi([n0_min, n0_max]);
    k0 = n0 * dk;
    
    % random amplitude (keep wave steepness a0*k0 in target range)
    a0 = sample_amplitude(a0_min, a0_max, k0, steepness_min, steepness_max);
    
    % generate eta and xi (single trig mode with known exact solution)
    om = sqrt(g*k0*tanh(k0*h));
    eta = a0*cos(k0*x);
    xi = a0*g*cosh(k0*(eta + h)).*sin(k0*x)./(om*cosh(k0*h));
    
    % compute exact DNO for this trig basis
    eta_x = myifft(1i*k.*myfft(eta,Nx));
    aux = k0*sinh(k0*(eta + h)).*sin(k0*x) - k0*eta_x.*cosh(k0*(eta + h)).*cos(k0*x);
    Gexac = a0*g*aux./(om*cosh(k0*h));
    
    % compute DNO with error-based truncation
    [Gxi, M_opt, err] = dno_adaptive(eta, xi, k, G0, Nx, M_max, Gexac);
    
    % store data
    eta_data(:, s) = eta;
    xi_data(:, s) = xi;
    Gxi_data(:, s) = Gxi;
    M_used(s) = M_opt;
    final_err(s) = err;
    
    fprintf('  Used M = %d terms, error = %.2e\n', M_opt, err);
end

% save training data
save('dno_training_data.mat', 'eta_data', 'xi_data', 'Gxi_data', 'M_used', ...
     'final_err', 'Nx', 'L', 'h', 'g', 'x', 'k');

fprintf('\nData generation complete. Saved to dno_training_data.mat\n');
fprintf('Average truncation order: %.1f\n', mean(M_used));
fprintf('Average final error: %.2e\n', mean(final_err));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [G, M_opt, err_best] = dno_adaptive(eta, xi, k, G0, Nx, M_max, Gexac)
% Computes DNO with error-based truncation
% Stops when error starts growing

etam = zeros(Nx, M_max+1);
etam(:,1) = ones(Nx, 1);
for m = 1:M_max
    etam(:,m+1) = multiply(eta, etam(:,m), Nx) ./ m;
end

fxi = myfft(xi, Nx);
xi_x = myifft(1i*k.*fxi);

% compute terms one by one, tracking error
Gm = zeros(Nx, M_max+1);
Gm(:,1) = myifft(G0.*fxi);

G = Gm(:,1);
norm_exact = norm(Gexac);
err_prev = norm(G - Gexac) / norm_exact;
err_best = err_prev;
M_opt = 0;
G_best = G;

for m = 1:M_max
    % compute m-th term
    Gm(:,m+1) = compute_Gm_term(m, etam, Gm, xi_x, k, G0, Nx);
    G = G + Gm(:,m+1);
    
    % compute error against exact solution
    err_curr = norm(G - Gexac) / norm_exact;
    
    % if error improved, update best
    if err_curr < err_best
        err_best = err_curr;
        M_opt = m;
        G_best = G;
    end
    
    % if error started growing, stop
    if err_curr > err_prev
        break;
    end
    
    err_prev = err_curr;
end

G = G_best;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Gm_term = compute_Gm_term(m, etam, Gm, xi_x, k, G0, Nx)
% Computes the m-th term in the DNO series

if mod(m, 2) == 0
    r = round(m/2);
    tmp = multiply(etam(:,m+1), xi_x, Nx);
    Gm_term = -myifft(G0.*k.^(2*(r-1)).*(1i*k).*myfft(tmp,Nx));
    for s = 0:r-1
        tmp = multiply(etam(:,2*(r-s)+1), Gm(:,2*s+1), Nx);
        Gm_term = Gm_term - myifft(k.^(2*(r-s)).*myfft(tmp,Nx));
        tmp = multiply(etam(:,2*(r-s)-1+1), Gm(:,2*s+1+1), Nx);
        Gm_term = Gm_term - myifft(G0.*k.^(2*(r-s-1)).*myfft(tmp,Nx));
    end
else
    r = round((m+1)/2);
    tmp = multiply(etam(:,m+1), xi_x, Nx);
    Gm_term = -myifft(k.^(2*(r-1)).*(1i*k).*myfft(tmp,Nx));
    for s = 0:r-2
        tmp = multiply(etam(:,2*(r-s)-1+1), Gm(:,2*s+1), Nx);
        Gm_term = Gm_term - myifft(G0.*k.^(2*(r-s-1)).*myfft(tmp,Nx));
        tmp = multiply(etam(:,2*(r-s-1)+1), Gm(:,2*s+1+1), Nx);
        Gm_term = Gm_term - myifft(k.^(2*(r-s-1)).*myfft(tmp,Nx));
    end
    tmp = multiply(etam(:,2), Gm(:,2*(r-1)+1), Nx);
    Gm_term = Gm_term - myifft(G0.*myfft(tmp,Nx));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [y] = multiply(a, b, Nx)
% Multiplies in physical space after Fourier extension

fac = 8;
Ny = fac*Nx;

fa = myfft(a, Nx);
fb = myfft(b, Nx);

fya = zeros(Ny, 1);
fyb = zeros(Ny, 1);

fya(1:Nx/2+1) = fa(1:Nx/2+1);
fyb(1:Nx/2+1) = fb(1:Nx/2+1);

fya(Ny+1-Nx/2:Ny) = fa(Nx/2+1:Nx);
fyb(Ny+1-Nx/2:Ny) = fb(Nx/2+1:Nx);

ya = myifft(fya);
yb = myifft(fyb);

fw = myfft(ya.*yb, Nx);

fy = zeros(Nx, 1);

fy(1:Nx/2+1) = fw(1:Nx/2+1);
fy(Nx/2+1:Nx) = fw(Ny+1-Nx/2:Ny);

y = fac*myifft(fy);

%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fy] = myfft(y, Nx)
% Direct Fourier transform

fy = fft(y);
fy(Nx/2+1) = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%

function [y] = myifft(fy)
% Inverse Fourier transform for real functions

y = real(ifft(fy));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function a0 = sample_amplitude(a0_min, a0_max, k0, steepness_min, steepness_max)
% Rejection sampling to enforce steepness_min <= k0 * a0 <= steepness_max

max_attempts = 1000;
for attempt = 1:max_attempts
    a0 = a0_min + (a0_max - a0_min)*rand();
    steepness = k0 * a0;
    if steepness_min <= steepness && steepness <= steepness_max
        return;
    end
end

a0_lower = max(a0_min, steepness_min / max(k0, eps));
a0_upper = min(a0_max, steepness_max / max(k0, eps));
if a0_lower > a0_upper
    error('No feasible amplitude range for k0 = %.6g.', k0);
end
warning('sample_amplitude: exceeded max attempts, sampling directly from feasible interval.');
a0 = a0_lower + (a0_upper - a0_lower)*rand();
