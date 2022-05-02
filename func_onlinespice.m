function [ theta_hat, clock_stop ] = func_onlinespice( y, H, L, type )
% Online sparse estimation using SPICE
% Dave Zachariah 2014-05-21
% Please cite if used

%Inputs:
% y    - Nx1 data vector
% H    - NxP regressor matrix
% L    - # iteration cycles per data sample
% type - 1 (real-valued case), else (complex-valued case)

%Outputs:
% theta_hat  - Px1 sparse parameter vector
% clock_stop - CPU time

%% Global variables
global P
global N
global bin_real

%% Initialize
clock_start = cputime;
[N,P]       = size(H);
Gamma       = zeros(P,P);
rho         = zeros(P,1);
kappa       = zeros(1,1);
theta_hat   = zeros(P,1);
bin_real    = type;

%% Process sample-by-sample
for n = 1:N
    [theta_hat, Gamma, rho, kappa ] = func_newsample( y(n), H(n,:), Gamma, rho, kappa, theta_hat, n, L );
    
    %TEMP:
   % disp(n)
   % plot(abs(theta_hat)), grid on
   % norm(theta_hat)
   % pause
end

%% Exit

clock_stop = cputime - clock_start;


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTIONS                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ theta_hat, Gamma, rho, kappa  ] = func_newsample( yn, h_row_n, Gamma, rho, kappa, theta_hat, n, L )

global P
global bin_real

%% Recursive update
Gamma = Gamma + h_row_n'*h_row_n;
rho   = rho + h_row_n'*yn;
kappa = kappa + abs(yn)^2;


%% Common variable
eta  = kappa + real(theta_hat'*Gamma*theta_hat) - 2*real( theta_hat'*rho );
zeta = rho - Gamma*theta_hat;


%% Cycle
for rep = 1:L
    for i = 1:P

        %Compute argument
        psi     = zeta(i) + Gamma(i,i)*theta_hat(i);
        phi_hat = angle(psi);
        
        %Compute alpha, beta, gamma
        alpha = eta + Gamma(i,i)*abs(theta_hat(i))^2 + 2*real( theta_hat(i)'*zeta(i) );
        beta  = real( Gamma(i,i) );
        gamma = abs( psi );

        %Update estimate
        theta_hat_i_new = 0;
        if (sqrt(n-1)*gamma > sqrt( alpha*beta - gamma^2 )) && (beta > 0)
            r_star          = (gamma/beta) - (1/beta) * sqrt( (alpha*beta - gamma^2)/(n-1) );
            theta_hat_i_new = r_star * exp(1j * phi_hat);
        end
        
        %Real-valued parameter
        if bin_real == 1
            theta_hat_i_new = real(theta_hat_i_new); %ensure numerically real-valued
        end
        
        %Update common variables
        eta  = eta + Gamma(i,i) * abs(theta_hat(i) - theta_hat_i_new )^2 + 2*real( (theta_hat(i) - theta_hat_i_new )'*zeta(i) );
        zeta = zeta + Gamma(:,i)*(theta_hat(i) - theta_hat_i_new ); 

        %Store update
        theta_hat(i) = theta_hat_i_new;

    end
end

%% Exit

end