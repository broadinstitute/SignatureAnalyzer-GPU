import tensorflow as tf

class NMF_algorithim:
    ''' implements ARD NMF from https://arxiv.org/pdf/1111.6085.pdf '''
    def __init__(self,Beta,H_prior,W_prior):
        # Beta paramaterizes the objective function
        # Beta = 1 induces a poisson objective
        # Beta = 2 induces a gaussian objective
        # Priors on the component matrices are Exponential (L1) and half-normal (L2)

        if Beta == 1 and H_prior == 'L1' and W_prior == 'L1' :
            self.update_W = update_W_poisson_L1
            self.update_H = update_H_poisson_L1
            self.lambda_update = update_lambda_L1

        elif Beta == 1 and H_prior == 'L1' and W_prior == 'L2':
            self.update_W = update_W_poisson_L2
            self.update_H = update_H_poisson_L1
            self.lambda_update = update_lambda_L2_L1

        elif Beta == 1 and H_prior == 'L2' and W_prior == 'L1':
            self.update_W = update_W_poisson_L1
            self.update_H = update_H_poisson_L2
            self.lambda_update = update_lambda_L1_L2

        elif Beta == 1 and H_prior == 'L2' and W_prior == 'L2':
            self.update_W = update_W_poisson_L2
            self.update_H = update_H_poisson_L2
            self.lambda_update = update_lambda_L2

        if Beta == 2 and H_prior == 'L1' and W_prior == 'L1':
            self.update_W = update_W_gaussian_L1
            self.update_H = update_H_gaussian_L1
            self.lambda_update = update_lambda_L1

        elif Beta == 2 and H_prior == 'L1' and W_prior == 'L2':
            self.update_W = update_W_gaussian_L2
            self.update_H = update_H_gaussian_L1
            self.lambda_update = update_lambda_L2_L1

        elif Beta == 2 and H_prior == 'L2' and W_prior == 'L1':
            self.update_W = update_W_gaussian_L1
            self.update_H = update_H_gaussian_L2
            self.lambda_update = update_lambda_L1_L2

        elif Beta == 2 and H_prior == 'L2' and W_prior == 'L2':
            self.update_W = update_W_gaussian_L2
            self.update_H = update_H_gaussian_L2
            self.lambda_update = update_lambda_L2

    def algorithm(self,W, H, V, lambda_, C, b0, eps_, phi):
        h_ = self.update_H(H, W, lambda_, phi, V, eps_)
        w_ = self.update_W(h_, W, lambda_, phi, V, eps_)
        lam_ = self.lambda_update(w_,h_,b0,C,eps_)
        return h_, w_,lam_


def t(X):
    return tf.transpose(X)


def beta_div(Beta,V,W,H,eps_):

    V_ap = tf.matmul(W, H) + eps_
    V = tf.convert_to_tensor(V, dtype=tf.float32)

    if Beta == 2:
        return tf.reduce_sum(tf.pow(V-V_ap,2))/2;
    if Beta == 1:
        lr = tf.log(tf.divide(V, V_ap))
        return tf.reduce_sum(tf.subtract(tf.add(tf.multiply(V, lr), V_ap), V))

def calculate_objective_function(Beta,V,W,H,lambda_,C, eps_,phi,K):
    loss = beta_div(Beta,V,W,H,eps_)
    K = tf.to_float(K)
    cst = tf.multiply(tf.multiply(K,C),1.0-tf.log(C))
    return tf.multiply(tf.pow(phi,-1),loss) + tf.multiply(C,tf.reduce_sum(tf.log(tf.multiply(lambda_,C)))) + cst

def update_H_poisson_L1(H, W, lambda_, phi, V, eps_):
    #beta = 1 gamma(beta) = 1
    denom = tf.reduce_sum(W, axis=0) + tf.divide(phi, lambda_) + eps_
    V_ap = tf.matmul(W, H) + eps_
    V_res = tf.divide(V, V_ap)
    # update = tf.divide(tf.transpose(tf.matmul(W, V_res,transpose_a=True)), denom)
    update = tf.divide(tf.matmul(W, V_res,transpose_a=True), tf.reshape(denom,[-1,1]))
    return tf.multiply(H,update)

def update_H_poisson_L2(H,W,lambda_,phi,V, eps_):
    #beta = 1 zeta(beta) = 1/2
    denom = tf.reshape(tf.reduce_sum(W, axis=0),[-1,1]) + tf.divide(phi*H, tf.reshape(lambda_,[-1,1])) + eps_
    V_ap = tf.matmul(W, H) + eps_
    V_res = tf.divide(V, V_ap)
    update = tf.pow(tf.divide(tf.matmul(W, V_res,transpose_a=True), denom),0.5)
    #update = tf.pow(tf.divide(tf.matmul(W, V_res,transpose_a=True), denom),0.5)
    return tf.multiply(H,update)

def update_H_gaussian_L1(H,W,lambda_,phi,V,eps_):
    #beta = 2 gamma(beta) = 1
    V_ap = tf.matmul(W, H) + eps_
    denom = tf.matmul(W,V_ap,transpose_a=True) + tf.reshape(tf.divide(phi, lambda_ ),[-1,1]) + eps_
    update = tf.divide(tf.matmul(W,V,transpose_a=True),denom)
    return tf.multiply(H,update)

def update_H_gaussian_L2(H,W,lambda_,phi,V,eps_):
    #beta = 2 zeta(beta) = 1
    V_ap = tf.matmul(W, H) + eps_
    denom = tf.matmul(W,V_ap,transpose_a=True) + tf.divide(phi * H, tf.reshape(lambda_,[-1,1]) ) + eps_
    update = tf.divide(tf.matmul(W,V,transpose_a=True),denom)
    return tf.multiply(H,update)

def update_W_poisson_L1(H, W, lambda_, phi, V, eps_):
    #beta = 1 gamma(beta) = 1
    denom = tf.reduce_sum(H, axis=1) + tf.divide(phi, lambda_ ) + eps_
    V_ap = tf.matmul(W, H) + eps_
    V_res = tf.divide(V, V_ap)
    update = tf.divide(tf.matmul(V_res, H,transpose_b=True), denom)
    return tf.multiply(W, update)

def update_W_poisson_L2(H,W,lambda_,phi,V,eps_):
    # beta = 1 zeta(beta) = 1/2
    V_ap = tf.matmul(W,H) + eps_
    V_res = tf.divide(V, V_ap)
    denom = tf.reduce_sum(H,axis=1) + tf.divide(phi*W,lambda_) + eps_
    update = tf.pow(tf.divide(tf.matmul(V_res,H,transpose_b=True),denom),0.5)
    return tf.multiply(W,update)

def update_W_gaussian_L1(H,W,lambda_,phi,V,eps_):
    #beta = 2 gamma(beta) = 1
    V_ap = tf.matmul(W,H) + eps_
    denom = tf.matmul(V_ap,t(H)) + tf.divide(phi,lambda_) + eps_
    update = tf.divide(tf.matmul(V,t(H)),denom)
    return tf.multiply(W,update)

def update_W_gaussian_L2(H,W,lambda_,phi,V,eps_):
    #beta = 2 zeta(beta) = 1
    V_ap = tf.matmul(W,H) + eps_
    denom = tf.matmul(V_ap,t(H)) + tf.divide(phi*W,lambda_) + eps_
    update = tf.divide(tf.matmul(V,t(H)),denom)
    return tf.multiply(W,update)

# update tolerance value for early stop criteria
def update_del(lambda_, lambda_last):
    del_ = tf.reduce_max(tf.divide(tf.abs(tf.subtract(lambda_, lambda_last)), lambda_last))
    return del_

# write new values to tensor
def assign_updates(Var):
    new_val = tf.placeholder(dtype=tf.float32)
    assignment = tf.assign(Var, new_val)
    return assignment, new_val

# write new values to H,W,Lambda
def apply_updates(H, W, Lambda):
    gen_H, h_prime = assign_updates(H)
    gen_W, w_prime = assign_updates(W)
    gen_Lambda, lambda_prime = assign_updates(Lambda)
    return gen_H, h_prime, gen_W, w_prime,gen_Lambda,lambda_prime

def update_lambda_L1(W,H,b0,C,eps_):
    return tf.divide(tf.reduce_sum(W, axis=0) + tf.reduce_sum(H, axis=1) + b0, C)

def update_lambda_L2(W,H,b0,C,eps_):
    return tf.divide(0.5*tf.reduce_sum(W*W, axis=0) + (0.5*tf.reduce_sum(H*H, axis=1))+b0,C)

def update_lambda_L1_L2(W,H,b0,C,eps_):
    return tf.divide(tf.reduce_sum(W, axis=0) + 0.5*tf.reduce_sum(H*H,axis=1)+b0,C)

def update_lambda_L2_L1(W,H,b0,C,eps_):
    return tf.divide(0.5*tf.reduce_sum(tf.pow(W,2), axis=0) + tf.reduce_sum(H,axis=1)+b0,C)

