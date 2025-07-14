import torch as torch
import torch.nn as nn
from torch.distributions.multivariate_normal import  MultivariateNormal
import numpy as np
import scipy
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Computes L_2 norm of the two lists of vectors x1 and x2
def custom_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))
    res.clamp_min_(0)
    return res

# Computes Gibbs kernel with a constant lengthscale and input dependent width
def f_kernel(Xdd,ell,σ1,σ2):
    prefactor = torch.outer(σ1,σ2)
    exponential = torch.exp(torch.clamp(-(Xdd)/(2*ell**2),min=(-100),max=0))
    K = prefactor*exponential
    return K

# Does radial DFT of a g(r) and send it to the corresponding S(q). Assumes Tailing behavior is 0.
def rdf2sq(r, rdf, q, ρ):
    dr = r[1] - r[0] # Stepsize
    sq   = torch.zeros(len(q))
    for j in range (len(q)):
        sq[j] = (4*np.pi*ρ*torch.trapz(r*(rdf)*torch.sin(q[j]*r),dx = dr)/q[j])
    return sq

# Does radial DFT on a set of g(r)s. Assumes Tailing behavior is 0.
def rdf2sq_batch(r, rdfs, q, ρ):
    sqs = torch.zeros((len(rdfs),len(q)))
    i = 0
    for rdf in rdfs:
        sqs[i] = torch.trapz(
                             (r*(rdf.reshape(len(r),1))).tile(1,len(q)).T*
                             (torch.sin(torch.outer(q.T[0],r.T[0]))/q.tile(1,len(r))),dx = float((r[1]-r[0])),dim=1)
        i += 1
    # Scale it by FT prefactor
    sqs *= 4*np.pi*ρ
    return sqs

# Does radial DFT of a S(q) and send it to the corresponding g(r). Assumes Tailing behavior is 0.
def sq2rdf(q, sq, r, ρ):
    dq = q[1] - q[0] # Stepsize
    rdf   = torch.zeros(len(r))
    for j in range (len(r)):
        rdf[j] = (1/(2*np.pi**2*r[j]))*torch.trapz(q*(sq)*torch.sin(r[j]*q)/ρ,dx = dq)
    return rdf

# Does radial DFT of a S(q) and send it to the corresponding g(r). Assumes Tailing behavior is 0.
def sq2rdf_batch(q, sqs, r, ρ):
    rdfs = torch.zeros((len(sqs),len(r)))
    i = 0
    for sq in sqs:
        rdfs[i] = torch.trapz(
                             (q*((sq).reshape(len(q),1))).tile(1,len(r)).T*
                             (torch.sin(torch.outer(r.T[0],q.T[0]))/r.tile(1,len(q))),dx = float((q[1]-q[0])),dim=1)/(ρ*2*np.pi**2)
        i += 1
    return rdfs

class nearestPDClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        # Symmetrize A to ensure it's symmetric
        A_sym = (A + A.t()) / 2

        # Check for PD to see if thats all it took
        if torch.distributions.constraints._PositiveDefinite().check(A_sym):
            return A_sym
        
        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)
        
        # Reconstruct the matrix
        A_pos_def = (eigenvectors @ torch.diag(eigenvalues)) @ eigenvectors.t()
        condition = torch.distributions.constraints._PositiveDefinite().check(A_pos_def)
        mineigval = torch.min(eigenvalues)
        i = 0
        while not condition:
            A_pos_def = eigenvectors @ torch.diag(eigenvalues + 1/(2*10**(15-i)) - mineigval) @ eigenvectors.t()
            i += 1
            condition = torch.distributions.constraints._PositiveDefinite().check(A_pos_def)
        return A_pos_def

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class data(Dataset):
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    if len(self.X) != len(self.Y):
      raise Exception("len(X) != len(Y)")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    _x = self.X[index].unsqueeze(dim=0)
    _y = self.Y[index].unsqueeze(dim=0)

    return _x, _y

def train_loop(dataloader, model, optimizer, totalEpochs, r_grid, q_train, sq_train, q_infer, r_infer, ylo_q, yhi_q, ylo_r, yhi_r):
    model.train()
    optimizer.zero_grad()
    losses = []
    tic = time.time()
    for epoch in range(totalEpochs):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            loss = model.NEG_LMLH_Trapz(r_grid, X[0], y[0])

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.detach().item())
            
        if epoch % 25 == 0:
            toc = time.time()
            average_loss = np.mean(np.array(losses[-int(len(dataloader)*10):]))
            print(f"Average loss: {average_loss:>7f}  [{epoch:>5d}/{totalEpochs:>5d}]")
            model.print_params()
            print(f"Minutes Taken Since Last Report: {(toc - tic)/60:>4f} ")
            print()

            with torch.no_grad():

                plt.title("S(q) Prediction")
                μ_q, Σ_q = model.predict_sq_trapz(r_grid, q_infer, q_train, sq_train)
                plt.scatter(q_train,sq_train,alpha=0.2,label="Presented Data")
                plt.plot(q_infer.detach().numpy(),μ_q.detach().numpy(),label="Mean Prediction")
                plt.fill_between(q_infer.T[0].detach().numpy(), μ_q.T[0].detach().numpy()+torch.diag(Σ_q).detach().numpy()**0.5, μ_q.T[0].detach().numpy()-torch.diag(Σ_q).detach().numpy()**0.5,alpha=0.5,label="1 +- std")
                plt.ylim(ylo_q,yhi_q)
                plt.xlim(q_infer[0],q_infer[-1])
                plt.legend()
                plt.show()
            
            tic = time.time()
            
            plt.figure(figsize=(6, 4))
            plt.plot(losses, label='Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Curve")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
    return losses


class GP(nn.Module):
    # Constructs GP object and establishes initial parameters of model
    def __init__(self, init_params, init_param_bounds, num_bonds, rho_init, SAMPLE_COUNT_init):
        super().__init__()

        self.construct_params(init_params,init_param_bounds)
        self.num_bonds = num_bonds
        self.nearestPD = nearestPDClass.apply
        self.rho = rho_init
        self.SAMPLE_COUNT = SAMPLE_COUNT_init

    def construct_params(self,init_params,init_params_bounds):
        # Ensure boundary conditions are met.
        for p in range(len(init_params)):
            if init_params[p] < init_params_bounds[p,0] or init_params[p] > init_params_bounds[p,1]:
                print("Parameter # " + str(p) + ", "  + str(init_params[p]) + ", is not in the range [" + str(init_params_bounds[p,0]) + ", " + str(init_params_bounds[p,1])  + "]")
                raise Exception("Initial parameter outside of chosen boundary.")

        # Create tensor representation of the parameters
        self.theta_raw = nn.Parameter(
         scipy.special.logit((init_params - init_params_bounds[:,0])/(init_params_bounds[:,1]-init_params_bounds[:,0])),requires_grad=True
        )

        self.theta_bounds = init_params_bounds
        self.theta = (self.theta_bounds[:,1] - self.theta_bounds[:,0])*torch.sigmoid(self.theta_raw) + self.theta_bounds[:,0]

    def compute_params_from_raw(self):
        self.theta = (self.theta_bounds[:,1] - self.theta_bounds[:,0])*torch.sigmoid(self.theta_raw) + self.theta_bounds[:,0]
        return self.theta
    
    def log_uniform_prior(self):
        sig = torch.sigmoid(self.theta_raw)
        return torch.sum(torch.log(sig) + torch.log1p(-sig))
 
    def print_params(self):
        params = self.compute_params_from_raw()
        print(f"l:       {params[0].item():>7f} ")
        print(f"max:     {params[1].item():>7f} ")
        print(f"slope:   {params[2].item():>7f} ")
        print(f"loc:     {params[3].item():>7f} ")
        print(f"decay:   {params[4].item():>7f} ")
        print(f"sigma_n: {params[5].item():>7f} ")
        print(f"r_0:     {params[6].item():>7f} ")
        print(f"s:       {params[7].item():>7f} ")
        for b in range(self.num_bonds):
            print(f"h:   {params[8+(b*3)].item():>7f} ")
            print(f"r:   {params[8+(b*3)+1].item():>7f} ")
            print(f"s:   {params[8+(b*3)+2].item():>7f} ")

    def K(self,r1,r2,ell,max,slope,loc,decay):
        Xdd = custom_cdist(r1,r2)
        Xdd_Reversed = custom_cdist(-r1,r2)
        σ1 = self.width_fxn(r1.T[0],max,slope,loc,decay)
        σ1_Reversed = self.width_fxn(-r1.T[0],max,slope,loc,decay)
        σ2 = self.width_fxn(r2.T[0],max,slope,loc,decay)
        return f_kernel(Xdd,ell,σ1,σ2) + f_kernel(Xdd_Reversed,ell,σ1_Reversed,σ2)
    
    def width_fxn(self,r,max,slope,loc,decay):
        return (max/(1+torch.exp(torch.clamp(-slope*(r-loc),min=-100,max=25))))*torch.exp(torch.clamp(decay*loc,min=-100,max=25))*torch.exp(torch.clamp(-r*decay,min=-100,max=25))
        
    def K_rr(self,r1,r2,adjust=True):
        Kdd = self.K(r1,r2,self.theta[0],self.theta[1],self.theta[2],self.theta[3],self.theta[4])
        if adjust:
            return self.nearestPD(Kdd)
        return Kdd

    def K_rq(self,r1,r2,q2,adjust=True):
        Kdd = self.K(r1,r2,self.theta[0],self.theta[1],self.theta[2],self.theta[3],self.theta[4])
        Krq = rdf2sq_batch(r2, Kdd, q2, self.rho)
        if adjust:
            return self.nearestPD(Krq)
        return Krq

    def K_qq(self,r1,r2,q1,q2,adjust=True):
        # Kdd = self.K(r1,r2,self.theta[0],self.theta[1],self.theta[2],self.theta[3],self.theta[4])
        Kdd = self.K_rr(r1,r2,adjust=True)
        Kqr = rdf2sq_batch(r2, Kdd, q2, self.rho)
        Kqq = rdf2sq_batch(r1, Kqr.T, q1, self.rho).T
        if adjust:
            return self.nearestPD(Kqq)
        return Kqq

    def mean_r(self,r):
        nonBond = 1/(1+torch.exp(torch.clamp(-self.theta[7]*(r-self.theta[6]),min=-20,max=20)))
        bond = torch.zeros_like(nonBond)
        for b in range(self.num_bonds):
            bond += (self.theta[8+(b*3)]/torch.sqrt(2*np.pi*self.theta[8+(b*3)+2]**2))*torch.exp(-0.5*(r-self.theta[8+(b*3)+1])**2/self.theta[8+(b*3)+2]**2)
        return nonBond + bond

    def mean_q(self,r,q):
        mean_r = self.mean_r(r)
        return rdf2sq(r.T[0], mean_r.T[0]-1, q.T[0], self.rho).unsqueeze(dim=1)

    def NEG_LMLH_Trapz(self, r_grid, q_train, sq_train):

        self.compute_params_from_raw()
        
        mu_q = self.mean_q(r_grid,q_train)
        Kdd = self.K_qq(r_grid,r_grid,q_train,q_train) + torch.eye(len(q_train))*(self.theta[5]**2)
        
        L = torch.linalg.cholesky(Kdd)
        logdet = torch.linalg.slogdet(Kdd)[1] 
        LMLH = 0.5*(sq_train - mu_q).reshape(1,-1) @ (torch.cholesky_solve(sq_train - mu_q,L)) + 0.5*logdet + (len(q_train)/2)*np.log(2*np.pi)
                
        return LMLH - self.log_uniform_prior()

    # This is a method I was trying before I figured out about the float64 things and was using a log GP
    # on the g(r) so it was strictly positive. It worked but was super slow and gave almost the same predictions
    # as a standard GP on g(r). The speed made it very unscaleable and very hard to tune the hyper parameters due to
    # big gradients.
    def NEG_LMLH_Monte_Carlo(self, r_grid, q_train, sq_train):

        self.compute_params_from_raw()

        mu_r = self.mean_r(r_grid)
        Kdd_r = self.K_rr(r_grid,r_grid)

        MVN_r = MultivariateNormal(mu_r.T[0],Kdd_r) 
        rsamples = MVN_r.rsample((self.SAMPLE_COUNT,)) # Must be rsample so we can backprop
        
        qsamples = rdf2sq_batch(r_grid, rsamples - 1, q_train, self.rho)
        
        mu_q = torch.mean(qsamples,dim=0).reshape(len(q_train),1)
        Kdd = torch.cov(qsamples.T) + torch.eye(len(q_train))*(self.theta[5]**2)
        
        L = torch.linalg.cholesky(Kdd)
        logdet = torch.linalg.slogdet(Kdd)[1] 
        LMLH = 0.5*(sq_train - mu_q).reshape(1,-1) @ (torch.cholesky_solve(sq_train - mu_q,L)) + 0.5*logdet + (len(q_train)/2)*np.log(2*np.pi)
        
        return LMLH

    def predict_sq_trapz(self, r_grid, q_infer, q_train, sq_train,adjust=True):
        self.compute_params_from_raw()
        
        Kii =  self.K_qq(r_grid,r_grid,q_infer,q_infer,adjust=adjust)
        Kdd =  self.K_qq(r_grid,r_grid,q_train,q_train,adjust=adjust)
        Kid =  self.K_qq(r_grid,r_grid,q_infer,q_train,adjust=False)
        Kdi =  Kid.T
        
        mu_q_dd = self.mean_q(r_grid,q_train)
        mu_q_ii = self.mean_q(r_grid,q_infer)
        
        Kdd = Kdd + torch.eye(len(q_train))*(self.theta[5]**2)
        
        L = torch.linalg.cholesky(Kdd)
        
        post_mean = mu_q_ii + Kid @ torch.cholesky_solve((sq_train - mu_q_dd).reshape(-1,1), L)
        post_cov = Kii - Kdi.T @ torch.cholesky_solve(Kdi, L)
        if adjust:
            return post_mean,  self.nearestPD((post_cov + post_cov.T)/2)
        return post_mean,  post_cov

    # This is a method I was trying before I figured out about the float64 things and was using a log GP
    # on the g(r) so it was strictly positive. It worked but was super slow and gave almost the same predictions
    # as a standard GP on g(r). The speed made it very unscaleable and very hard to tune the hyper parameters due to
    # big gradients.
    def predict_sq_monte_carlo(self, r_grid, q_infer, q_train, sq_train, optimal_shrinkage):
        self.compute_params_from_raw()
        
        q = torch.zeros((len(q_infer)+len(q_train)),1)
        
        q[:len(q_infer)] = q_infer
        q[len(q_infer):] = q_train
        
        mu_r = self.mean_r(r_grid)
        Kdd_r = self.K_rr(r_grid,r_grid)

        MVN_r = MultivariateNormal(mu_r.T[0],Kdd_r) 
        rsamples = MVN_r.rsample((self.SAMPLE_COUNT,)).double() # Must be rsample so we can backprop
        
        qsamples = rdf2sq_batch(r_grid, rsamples-1, q, self.rho)

        if optimal_shrinkage:

            cov = torch.cov(qsamples.T)

            mean = []
            var = []
            std = []
            skews = []
            kurtoses = []
            
            for index in range(len(q)):
                array = qsamples.T[index]
                mean.append(torch.mean(array))
                diffs = array - torch.mean(array)
                var.append(torch.mean(torch.pow(diffs, 2.0)))
                std.append(torch.pow(torch.mean(torch.pow(diffs, 2.0)), 0.5))
                zscores = diffs / torch.pow(torch.mean(torch.pow(diffs, 2.0)), 0.5)
                skews.append(torch.mean(torch.pow(zscores, 3.0)))
                kurtoses.append(torch.mean(torch.pow(zscores, 4.0)) - 3.0) 
            
            elipitical_kurtoses = torch.mean(torch.tensor(kurtoses))/3
            print("Eliptical Kurtoses:", elipitical_kurtoses)
            
            mean_ev = (torch.trace(cov @ cov)/self.SAMPLE_COUNT) - (1+elipitical_kurtoses)*(self.SAMPLE_COUNT/len(cov))*(torch.trace(cov)/self.SAMPLE_COUNT)**2
            
            print("Mean of Eigenvalues:", mean_ev)
            
            sphericity = mean_ev/((torch.trace(cov)/self.SAMPLE_COUNT)**2)
            
            print("Sphericity:",sphericity)
            print()
            
            β_0 = (sphericity-1)/((sphericity-1) + elipitical_kurtoses*(2*sphericity + self.SAMPLE_COUNT)/len(cov) + (sphericity +  self.SAMPLE_COUNT)/(len(cov) -1))
            α_0 = (1-β_0)*torch.trace(cov)/self.SAMPLE_COUNT
            
            print("α_0:", α_0)
            print("β_0:", β_0)
            
            S_beta_alpha = β_0*cov +α_0*torch.eye(len(cov))
            print("isPD(S_beta_alpha):",torch.distributions.constraints._PositiveDefinite().check(S_beta_alpha))
            print("Frob Norm of Old Estimate with New:",torch.norm(S_beta_alpha - cov))
            print()
            
            cov = self.nearestPD(S_beta_alpha)

        else:
            cov = self.nearestPD(torch.cov(qsamples.T))

        Kii = cov[:len(q_infer),:len(q_infer)]
        Kdd = cov[len(q_infer):,len(q_infer):]
        Kid = cov[:len(q_infer),len(q_infer):]
        Kdi = cov[len(q_infer):,:len(q_infer)]
        
        mu_q_dd = torch.mean(qsamples[:,len(q_infer):],dim=0).reshape(len(q_train),1)
        mu_q_ii = torch.mean(qsamples[:,:len(q_infer)],dim=0).reshape(len(q_infer),1)
        
        Kdd = Kdd + torch.eye(len(q_train))*(self.theta[5]**2)
        
        L = torch.linalg.cholesky(Kdd)
        
        post_mean = mu_q_ii + Kid @ torch.cholesky_solve((sq_train - mu_q_dd).reshape(-1,1), L)
        post_cov = Kii - Kdi.T @ torch.cholesky_solve(Kdi, L)
        
        return post_mean,  self.nearestPD((post_cov + post_cov.T)/2)

    def predict_rdf_trapz(self, r_grid, r_infer, q_train, sq_train,adjust=True): 
        self.compute_params_from_raw()
        
        Kii =  self.K_rr(r_infer,r_infer,adjust=adjust)
        Kdd =  self.K_qq(r_grid,r_grid,q_train,q_train,adjust=adjust)
        Kid =  self.K_rq(r_infer,r_grid,q_train,adjust=False)
        Kdi =  Kid.T
        
        mu_q_dd = self.mean_q(r_grid,q_train)
        mu_r_ii = self.mean_r(r_infer)
        
        Kdd = Kdd + torch.eye(len(q_train))*(self.theta[5]**2)
        
        L = torch.linalg.cholesky(Kdd)
        
        post_mean = mu_r_ii + Kid @ torch.cholesky_solve((sq_train - mu_q_dd).reshape(-1,1), L)
        post_cov = Kii - Kdi.T @ torch.cholesky_solve(Kdi, L)

        if adjust:
            return post_mean,  self.nearestPD((post_cov + post_cov.T)/2)
            
        return post_mean,  post_cov

    # This is a method I was trying before I figured out about the float64 things and was using a log GP
    # on the g(r) so it was strictly positive. It worked but was super slow and gave almost the same predictions
    # as a standard GP on g(r). The speed made it very unscaleable and very hard to tune the hyper parameters due to
    # big gradients.
    def predict_rdf_monte_carlo(self, r_grid, r_infer, q_infer, q_train, sq_train, optimal_shrinkage,return_untouched_data=False):
        self.compute_params_from_raw()
        
        q = torch.zeros((len(q_infer)+len(q_train)),1)
        
        q[:len(q_infer)] = q_infer
        q[len(q_infer):] = q_train
        
        mu_r = self.mean_r(r_grid)
        Kdd_r = self.K_rr(r_grid,r_grid)

        MVN_r = MultivariateNormal(mu_r.T[0],Kdd_r) 
        rsamples = MVN_r.rsample((self.SAMPLE_COUNT,)).double() # Must be rsample so we can backprop
        
        qsamples = rdf2sq_batch(r_grid, rsamples-1, q, self.rho)

        if optimal_shrinkage:

            cov = torch.cov(qsamples.T)

            mean = []
            var = []
            std = []
            skews = []
            kurtoses = []
            
            for index in range(len(q)):
                array = qsamples.T[index]
                mean.append(torch.mean(array))
                diffs = array - torch.mean(array)
                var.append(torch.mean(torch.pow(diffs, 2.0)))
                std.append(torch.pow(torch.mean(torch.pow(diffs, 2.0)), 0.5))
                zscores = diffs / torch.pow(torch.mean(torch.pow(diffs, 2.0)), 0.5)
                skews.append(torch.mean(torch.pow(zscores, 3.0)))
                kurtoses.append(torch.mean(torch.pow(zscores, 4.0)) - 3.0) 
            
            elipitical_kurtoses = torch.mean(torch.tensor(kurtoses))/3
            print("Eliptical Kurtoses:", elipitical_kurtoses)
            
            mean_ev = (torch.trace(cov @ cov)/self.SAMPLE_COUNT) - (1+elipitical_kurtoses)*(self.SAMPLE_COUNT/len(cov))*(torch.trace(cov)/self.SAMPLE_COUNT)**2
            
            print("Mean of Eigenvalues:", mean_ev)
            
            sphericity = mean_ev/((torch.trace(cov)/self.SAMPLE_COUNT)**2)
            
            print("Sphericity:",sphericity)
            print()
            
            β_0 = (sphericity-1)/((sphericity-1) + elipitical_kurtoses*(2*sphericity + self.SAMPLE_COUNT)/len(cov) + (sphericity +  self.SAMPLE_COUNT)/(len(cov) -1))
            α_0 = (1-β_0)*torch.trace(cov)/self.SAMPLE_COUNT
            
            print("α_0:", α_0)
            print("β_0:", β_0)
            
            S_beta_alpha = β_0*cov +α_0*torch.eye(len(cov))
            print("isPD(S_beta_alpha):",torch.distributions.constraints._PositiveDefinite().check(S_beta_alpha))
            print("Frob Norm of Old Estimate with New:",torch.norm(S_beta_alpha - cov))
            print()
            
            cov = self.nearestPD(S_beta_alpha)

            S_beta_alpha_q = S_beta_alpha

        else:
            cov = self.nearestPD(torch.cov(qsamples.T))

        Kii = cov[:len(q_infer),:len(q_infer)]
        Kdd = cov[len(q_infer):,len(q_infer):]
        Kid = cov[:len(q_infer),len(q_infer):]
        Kdi = cov[len(q_infer):,:len(q_infer)]
        
        mu_q_dd = torch.mean(qsamples[:,len(q_infer):],dim=0).reshape(len(q_train),1)
        mu_q_ii = torch.mean(qsamples[:,:len(q_infer)],dim=0).reshape(len(q_infer),1)
        
        Kdd = Kdd + torch.eye(len(q_train))*(self.theta[5]**2)
        
        L = torch.linalg.cholesky(Kdd)
        
        post_mean = mu_q_ii + Kid @ torch.cholesky_solve((sq_train - mu_q_dd).reshape(-1,1), L)
        post_cov = Kii - Kdi.T @ torch.cholesky_solve(Kdi, L)

        MVN_post_q = MultivariateNormal(post_mean.T[0],  self.nearestPD((post_cov + post_cov.T)/2))
        qsamples_post = MVN_post_q.rsample((self.SAMPLE_COUNT,)).double()
        
        rsamples_post = sq2rdf_batch(q_infer, qsamples_post, r_infer, self.rho)
        
        if optimal_shrinkage:

            print("Computing real space post using optimal shrinkage.")

            cov = torch.cov(rsamples_post.T)

            mean = []
            var = []
            std = []
            skews = []
            kurtoses = []
            
            for index in range(len(r_infer)):
                array = rsamples_post.T[index]
                mean.append(torch.mean(array))
                diffs = array - torch.mean(array)
                var.append(torch.mean(torch.pow(diffs, 2.0)))
                std.append(torch.pow(torch.mean(torch.pow(diffs, 2.0)), 0.5))
                zscores = diffs / torch.pow(torch.mean(torch.pow(diffs, 2.0)), 0.5)
                skews.append(torch.mean(torch.pow(zscores, 3.0)))
                kurtoses.append(torch.mean(torch.pow(zscores, 4.0)) - 3.0) 
            
            elipitical_kurtoses = torch.mean(torch.tensor(kurtoses))/3
            print("Eliptical Kurtoses:", elipitical_kurtoses)
            
            mean_ev = (torch.trace(cov @ cov)/self.SAMPLE_COUNT) - (1+elipitical_kurtoses)*(self.SAMPLE_COUNT/len(cov))*(torch.trace(cov)/self.SAMPLE_COUNT)**2
            
            print("Mean of Eigenvalues:", mean_ev)
            
            sphericity = mean_ev/((torch.trace(cov)/self.SAMPLE_COUNT)**2)
            
            print("Sphericity:",sphericity)
            print()
            
            β_0 = (sphericity-1)/((sphericity-1) + elipitical_kurtoses*(2*sphericity + self.SAMPLE_COUNT)/len(cov) + (sphericity +  self.SAMPLE_COUNT)/(len(cov) -1))
            α_0 = (1-β_0)*torch.trace(cov)/self.SAMPLE_COUNT
            
            print("α_0:", α_0)
            print("β_0:", β_0)
            
            S_beta_alpha = β_0*cov +α_0*torch.eye(len(cov))
            print("isPD(S_beta_alpha):",torch.distributions.constraints._PositiveDefinite().check(S_beta_alpha))
            print("Frob Norm of Old Estimate with New:",torch.norm(S_beta_alpha - cov))
            print()
            
            cov = self.nearestPD(S_beta_alpha)

            S_beta_alpha_r = S_beta_alpha

        else:
            cov = self.nearestPD(torch.cov(rsamples_post.T))

        mean = torch.mean(rsamples_post,dim=0).reshape(len(r_infer),1)

        if return_untouched_data:
            return  mean, self.nearestPD((cov + cov.T)/2), S_beta_alpha_q, S_beta_alpha_r

        return mean, self.nearestPD((cov + cov.T)/2)

        
        