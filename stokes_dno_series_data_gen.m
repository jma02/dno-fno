function stokes_dno_gen_data

% Generates time-dependent (eta, xi) -> G(eta)xi data using Stokes waves
% xi is taken as the free-surface velocity potential trace (phi in stokes.m)

clc; clear all;

% numerical parameters
%%%%%%%%%%%%%%%%%%%%%%

Nx = 1024;
L = 164;

dx = L/Nx;
dk = 2*pi/L;
x = dx*(0:Nx-1)';
k = dk*[0:Nx/2,1-Nx/2:-1]';

% time parameters
%%%%%%%%%%%%%%%%%

dt = 0.1;
Tmax = 20;
t = 0:dt:Tmax;
Nt = length(t);

% physical parameters
%%%%%%%%%%%%%%%%%%%%%

g = 1;
h = 1;

% data generation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nsamples = 392;           % number of parameter draws (a0, n0)
a0_min = 0.02;
a0_max = 0.30;
steepness_max = 0.15;    % maximum k0*a0

% per-regime wavenumber ranges
%   deep water: n0 in {1,...,20}  (all k0h effectively infinite)
%   finite depth: n0 in {14,...,26}  (k0h in [0.54, 1.00], intermediate depth)
n0_min_deep = 1;   n0_max_deep = 20;
n0_min_finite = 14; n0_max_finite = 26;

M = 6;                   % DNO Taylor series truncation order
nprint = 50;             % number of saved datasets
iprint = max(1, floor(Nt/nprint));

save_idx = unique([1:iprint:Nt, Nt]);
Nsaves = length(save_idx);

rng(42);
ichoi_values = [0, 1];

for regime = 1:length(ichoi_values)
    ichoi = ichoi_values(regime);

    % storage for training data (per regime)
    eta_data = zeros(Nx, Nsaves, Nsamples);
    xi_data = zeros(Nx, Nsaves, Nsamples);
    Gxi_data = zeros(Nx, Nsaves, Nsamples);

    params = struct('a0', zeros(Nsamples,1), 'n0', zeros(Nsamples,1), ...
        'k0', zeros(Nsamples,1), 'ichoi', ichoi*ones(Nsamples,1));

    fprintf('\n=== Generating data for ichoi = %d (0=deep,1=finite) ===\n', ichoi);

    for s = 1:Nsamples
        fprintf('Sample %d/%d\n', s, Nsamples);

        if ichoi == 0
            n0 = randi([n0_min_deep, n0_max_deep]);
        else
            n0 = randi([n0_min_finite, n0_max_finite]);
        end
        k0 = n0*dk;
        a0 = sample_amplitude(a0_min, a0_max, k0, steepness_max);

        params.a0(s) = a0;
        params.n0(s) = n0;
        params.k0(s) = k0;

    % derived parameters
    eps = k0*a0;
    eps2 = eps^2;
    eps3 = eps^3;
    eps4 = eps^4;

    sigma = tanh(k0*h);
    alfa1 = cosh(2*k0*h);

    om0 = sqrt(g*k0*sigma);
    om2 = 0.25*(2*alfa1^2 + 7)/(alfa1 - 1)^2;
    om4 = (20*alfa1^5 + 112*alfa1^4 - 100*alfa1^3 - 68*alfa1^2 - 211*alfa1 + 328)/(32*(alfa1 - 1)^5);

    om = om0*(1 + eps2*om2 + eps4*om4);

    B31 = (3 + 8*sigma^2 - 9*sigma^4)/(16*sigma^4);
    B51 = (121*alfa1^5 + 263*alfa1^4 + 376*alfa1^3 - 1999*alfa1^2 + 2509*alfa1 - 1108)/(192*(alfa1 - 1)^5);
    B22 = 0.25*(3 - sigma^2)/sigma^3;
    denom = 24*sinh(2*k0*h)*(3*alfa1 + 2)*(alfa1 - 1)^4;
    B42 = (60*alfa1^6 + 232*alfa1^5 - 118*alfa1^4 - 989*alfa1^3 - 607*alfa1^2 + 352*alfa1 + 260)/denom;
    B33 = (27 - 9*sigma^2 + 9*sigma^4 - 3*sigma^6)/(64*sigma^6);
    denom = 128*(3*alfa1 + 2)*(alfa1 - 1)^6;
    B53 = 9*(57*alfa1^7 + 204*alfa1^6 - 53*alfa1^5 - 782*alfa1^4 - 741*alfa1^3 - 52*alfa1^2 + 371*alfa1 + 186)/denom;
    denom = 24*sinh(2*k0*h)*(3*alfa1 + 2)*(alfa1 - 1)^4;
    B44 = (24*alfa1^6 + 116*alfa1^5 + 214*alfa1^4 + 188*alfa1^3 + 133*alfa1^2 + 101*alfa1 + 34)/denom;
    denom = 384*(12*alfa1^2 + 11*alfa1 + 2)*(alfa1 - 1)^6;
    B55 = 5*(300*alfa1^8 + 1579*alfa1^7 + 3176*alfa1^6 + 2949*alfa1^5 + 1188*alfa1^4 + 675*alfa1^3 + 1326*alfa1^2 + 827*alfa1 + 130)/denom;

    A11 = 1/sinh(k0*h);
    A31 = 0;
    A51 = 0;
    A22 = 3/(8*sinh(k0*h)^4);
    A42 = (12*alfa1^4 + 22*alfa1^3 - 84*alfa1^2 - 135*alfa1 + 104)/(24*(alfa1 - 1)^5);
    A33 = (9 - 4*sinh(k0*h)^2)/(64*sinh(k0*h)^7);
    denom = 64*sinh(k0*h)*(3*alfa1 + 2)*(alfa1 - 1)^6;
    A53 = (8*alfa1^6 + 138*alfa1^5 + 384*alfa1^4 - 568*alfa1^3 - 2388*alfa1^2 + 237*alfa1 + 974)/denom;
    denom = 48*(3*alfa1 + 2)*(alfa1 - 1)^5;
    A44 = (10*alfa1^3 - 174*alfa1^2 + 291*alfa1 + 278)/denom;
    denom = 64*sinh(k0*h)*(3*alfa1 + 2)*(4*alfa1 + 1)*(alfa1 - 1)^6;
    A55 = (-6*alfa1^5 + 272*alfa1^4 - 1552*alfa1^3 + 852*alfa1^2 + 2029*alfa1 + 430)/denom;
    C2 = 0.25*(sigma^2 - 1)/sigma;
    C4 = -9/(4*sinh(2*k0*h)*(alfa1 - 1)^3);

        if ichoi == 0
            h_use = 1000;
            om0 = sqrt(g*k0);
            om = om0*(1 + 0.5*eps2 + 5*eps4/8);
        else
            h_use = h;
        end

        G0 = k.*tanh(h_use*k);

        for si = 1:Nsaves
            iter = save_idx(si);
            theta = k0*x - om*t(iter);

            eta = (1 + eps2*B31 + eps4*B51)*cos(theta) + eps*(B22 + eps2*B42)*cos(2*theta) ...
                  + eps2*(B33 + eps2*B53)*cos(3*theta) + eps3*B44*cos(4*theta) + eps4*B55*cos(5*theta);
            eta = a0*eta;

            phi = (A11 + eps2*A31 + eps4*A51)*cosh(k0*(eta + h)).*sin(theta) ...
                  + eps*(A22 + eps2*A42)*cosh(2*k0*(eta + h)).*sin(2*theta) ...
                  + eps2*(A33 + eps2*A53)*cosh(3*k0*(eta + h)).*sin(3*theta) ...
                  + eps3*A44*cosh(4*k0*(eta + h)).*sin(4*theta) ...
                  + eps4*A55*cosh(5*k0*(eta + h)).*sin(5*theta) ...
                  + (eps*C2 + eps3*C4)*om0*t(iter)/sigma;
            phi = a0*om0*phi./k0;

            if ichoi == 0
                eta = (1 + eps2/8 + 121*eps4/192)*cos(theta) + (0.5*eps + 5*eps3/6)*cos(2*theta) ...
                      + (3*eps2/8 + 171*eps4/128)*cos(3*theta) + eps3*cos(4*theta)./3 ...
                      + 125*eps4*cos(5*theta)./384;
                eta = a0*eta;

                phi = exp(k0*eta).*sin(theta) + 0.5*eps3*exp(2*k0*eta).*sin(2*theta) ...
                      + eps4*exp(3*k0*eta).*sin(3*theta)./12;
                phi = a0*om0*phi./k0;
            end

            xi = phi;
            Gxi = dno_series_eval(eta, xi, k, G0, Nx, M);

            eta_data(:, si, s) = eta;
            xi_data(:, si, s) = xi;
            Gxi_data(:, si, s) = Gxi;
        end
    end

    outfile = sprintf('stokes_dno_training_ichoi%d.mat', ichoi);
    save(outfile, 'eta_data', 'xi_data', 'Gxi_data', 'params', ...
        'x', 'k', 't', 'save_idx', 'Nx', 'L', 'h', 'g', 'dt', 'Tmax', 'M', 'ichoi');

    fprintf('\nSaved regime data to %s\n', outfile);
end

fprintf('\nData generation complete for both regimes.\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [G] = dno_series_eval(eta, xi, k, G0, Nx, M)
% Sums up all terms to evaluate the DNO series

etam = zeros(Nx, M+1);

etam(:,1) = ones(Nx, 1);
for m = 1:M
    etam(:,m+1) = multiply(eta, etam(:,m), Nx) ./ m;
end

fxi = myfft(xi, Nx);
xi_x = myifft(1i*k.*fxi);

Gm = zeros(Nx, M+1);
Gm(:,1) = myifft(G0.*fxi);

for m = 1:M
    Gm(:,m+1) = compute_Gm_term(m, etam, Gm, xi_x, k, G0, Nx);
end

G = sum(Gm, 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Gm_term = compute_Gm_term(m, etam, Gm, xi_x, k, G0, Nx)
% Computes the m-th term in the DNO series

if mod(m, 2) == 0
    r = round(m/2);
    tmp = multiply(etam(:,m+1), xi_x, Nx);
    Gm_term = -myifft(G0.*k.^(2*(r-1)).*(1i*k).*myfft(tmp, Nx));
    for s = 0:r-1
        tmp = multiply(etam(:,2*(r-s)+1), Gm(:,2*s+1), Nx);
        Gm_term = Gm_term - myifft(k.^(2*(r-s)).*myfft(tmp, Nx));
        tmp = multiply(etam(:,2*(r-s)-1+1), Gm(:,2*s+1+1), Nx);
        Gm_term = Gm_term - myifft(G0.*k.^(2*(r-s-1)).*myfft(tmp, Nx));
    end
else
    r = round((m+1)/2);
    tmp = multiply(etam(:,m+1), xi_x, Nx);
    Gm_term = -myifft(k.^(2*(r-1)).*(1i*k).*myfft(tmp, Nx));
    for s = 0:r-2
        tmp = multiply(etam(:,2*(r-s)-1+1), Gm(:,2*s+1), Nx);
        Gm_term = Gm_term - myifft(G0.*k.^(2*(r-s-1)).*myfft(tmp, Nx));
        tmp = multiply(etam(:,2*(r-s-1)+1), Gm(:,2*s+1+1), Nx);
        Gm_term = Gm_term - myifft(k.^(2*(r-s-1)).*myfft(tmp, Nx));
    end
    tmp = multiply(etam(:,2), Gm(:,2*(r-1)+1), Nx);
    Gm_term = Gm_term - myifft(G0.*myfft(tmp, Nx));
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

function a0 = sample_amplitude(a0_min, a0_max, k0, steepness_max)
% Rejection sampling to enforce k0 * a0 <= steepness_max

max_attempts = 1000;
for attempt = 1:max_attempts
    a0 = a0_min + (a0_max - a0_min)*rand();
    if k0 * a0 <= steepness_max
        return;
    end
end

warning('sample_amplitude: exceeded max attempts, clamping to steepness limit.');
a0 = min(a0_max, steepness_max / max(k0, eps));
