%% Estimation of graph using OLspice for Block diagonal matrix
% Arun Venkitaraman  2017-11-26
close all
clear all
clc
p=6;
niter=20;


%% Cycle graph
B=.5*circshift(eye(p),1)+0.0*circshift(eye(p),-2)+0.0*circshift(eye(p),-2); % AR-1 graph
Lam=1*diag(rand(1,p)); % unequal variances
%Lam=eye(p); % Equal variances
B=B+B';
%B=B/2;
B(1,p)=0;
B(p,1)=0;

B= [0 0 0 0 .9*1 0;
    0 0 0 .9*1 0 0;
    0 .5*1 0 0 0 .5*1;
    .9 0 0 0 (0) 0;
    (0) 0 0 0 0 .75*1;
    0 0 0 .5*1 0 0];


p=5;
B3=((sprand(p,p,.4))); % sparse random B matrix
B4=((sprand(p,p,.4))); % sparse random B matrix

B=[B3 zeros(p);
    zeros(p) B4];

%load('Bfeas_sbm_symm.mat');
p=size(B,2);
Lam=1*diag(rand(1,p));
%B=B+B';
B=B-diag(diag(B));
Sigma=pinv(eye(p)-0*B)*Lam*(pinv(eye(p)-0*B))'; % Covariance
mu=zeros(p,1)';

B0=.5*(B+B');
nsamp_vec=round(p*logspace(0,2,5));
nlen=50+0*length(nsamp_vec);
p=5;

%clear B0;
%M=load('Bfeas_sbm.mat');
load('Bfeas_sbm_symm.mat');

B0=[B0(1:5,1:5) zeros(5);
    zeros(5) B0(6:10,6:10)];

B0(5,10)=.25;
B0(10,5)=.25;
B0(1,6)=.25;
B0(6,1)=.25;
B0=.5*(B0+B0);
%B0=B0/max(B0(:));
B=B0;
p=10;
%B0=M.B0;
%B=M.B0;
for ni=1:length(nsamp_vec)%%
    msedel1=zeros(nlen,1);
    msedel2=zeros(nlen,1);
    msedel3=zeros(nlen,1);
    Mse1=zeros(nlen,1);
    Mse2=zeros(nlen,1);
    Mse3=zeros(nlen,1);
    mse_gr=zeros(nlen,1);
    e_b=zeros(nlen,1);
    n_del=zeros(nlen,1);
    En=zeros(nlen,1);
    for s=1:nlen
        n=nsamp_vec(ni);
        
        e=mvnrnd(mu,Lam,n); % Samples generated
        xtrain=(eye(p)-B)\e';
        xtrain=xtrain';
        
        X=xtrain;
        
        Bhat=zeros(p);
        
        % OL-SPICE iterations
        
        for nod=1:p
            
            y=X(:,nod);
            H=X(:,[1:nod-1 nod+1:p]);
            
            
            %[theta_hat, ~] = func_onlinespice(y,H,niter,1);
            theta_hat=zeros(p,1);
            theta_hat=(theta_hat');
            
            %theta_hat_ls=pinv(H)*y;
            theta_hat_ls=inv(H'*H+1*eye(p-1))*H'*y;
            theta_hat_ls=(theta_hat_ls');
            
            Bhat(nod,:)=[theta_hat(1:nod-1) 0 theta_hat(nod:p-1)];
            Bhat_ls(nod,:)=[theta_hat_ls(1:nod-1) 0 theta_hat_ls(nod:p-1)];
        end
        mse_gr(s)=norm(B-Bhat,'fro')^2; % In estimation of matrix
        mse_gr_ls(s)=norm(B-Bhat_ls,'fro')^2; % In estimation of matrix
        e_b(s)=norm(B,'fro')^2;
        
        %% Prediction results
        %nodes_a=[1:2:p];
        %nodes_a=[1:2:p];
        nodes_b=[2 4 6 8 10];
        nodes_a=setdiff(1:p,nodes_b);
        
        e=mvnrnd(mu,Lam,10000); % Samples generated
        xtest=(eye(p)-B)\(e');
        xtest=xtest';
        
        x_b=xtest(:,nodes_b)';
        x_a=xtest(:,nodes_a)';
        
        xhat_a1=graphPrediction(nodes_a,nodes_b,B0,x_b);
        xhat_a2=graphPrediction(nodes_a,nodes_b,Bhat,x_b);
        xhat_a3=graphPrediction(nodes_a,nodes_b,Bhat_ls,x_b);
        
        Mse1(s)=(norm(x_a-xhat_a1,'fro')^2);
        Mse2(s)=(norm(x_a-xhat_a2,'fro')^2);
        Mse3(s)=(norm(x_a-xhat_a3,'fro')^2);
        
        En(s)=norm(x_a,'fro')^2;
        
        %
        %% Filtering results
        
        
        
        %P=[8 9 10];
        %P=[1 10];
          P=[2 6 10];
        e=mvnrnd(mu,Lam,10000); % Samples generated
        del=zeros(p,10000);
        ed=e';
        del(P,:)=10*ed(P,:);
        xtest=(eye(p)-B)\(e'+del);
        xtest=xtest';
        
        delhat1=(eye(p)-B0)*xtest';
        msedel1(s)=(norm(del-delhat1,'fro')^2);
        delhat2=(eye(p)-Bhat)*xtest';
        msedel2(s)=(norm(del-delhat2,'fro')^2);
        n_del(s)=norm(del,'fro')^2;
        delhat3=(eye(p)-Bhat_ls)*xtest';
        msedel3(s)=(norm(del-delhat3,'fro')^2);
        
    end
    mse_pred(ni,:)=10*log10([mean(Mse1)/mean(En) mean(Mse2)/mean(En) mean(Mse3)/mean(En)]);
    mse_filt(ni,:)=10*log10([mean(msedel1)/mean(n_del) mean(msedel2)/mean(n_del) mean(msedel3)/mean(n_del)]);
    mse_B(ni,:)=10*log10([mean(mse_gr)/mean(e_b) mean(mse_gr_ls)/mean(e_b)]);
end

figure, plot(nsamp_vec,mse_filt), xlabel('N'),ylabel('NFE'), legend('True','SPICE estimate','LS estimate');
figure, plot(nsamp_vec,mse_pred),xlabel('N'),ylabel('NPE'),legend('True','SPICE estimate','LS estimate');
figure, plot(nsamp_vec,mse_B),xlabel('N'),ylabel('NMSE'),legend('SPICE estimate','LS estimate');


