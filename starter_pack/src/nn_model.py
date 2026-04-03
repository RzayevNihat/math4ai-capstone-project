import numpy as np

class OneHiddenLayerNN:
    def __init__(self,input_dim,hidden_dim,output_dim,reg_lambda=1e-4,seed=52):
        rng=np.random.default_rng(seed)
        self.W1=rng.normal(0,0.01,(hidden_dim,input_dim))
        self.b1=np.zeros((1,hidden_dim))
        self.W2=rng.normal(0,0.01,(output_dim,hidden_dim))
        self.b2=np.zeros((1,output_dim))
        self.reg_lambda=reg_lambda
        self.velocity = None
        self.m = None
        self.v = None
        self.t = 0
    
    def tanh(self,z):
        h=np.tanh(z)
        return h
    def tanh_grad(self,h):
        return 1- h**2
    def softmax(self,z):
        z=z-np.max(z,axis=1,keepdims=True)
        exp=np.exp(z)
        return exp/np.sum(exp,axis=1,keepdims=True)
    def forward(self,X):
        self.X=X
        self.z1=X @ self.W1.T + self.b1
        self.H=self.tanh(self.z1)
        self.z2= self.H @ self.W2.T + self.b2
        self.P=self.softmax(self.z2)
        return self.P
    def compute_loss(self, y):
        N = y.shape[0]
        probs = self.P[np.arange(N), y]
        reg_loss=(self.reg_lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        data_loss=-np.mean(np.log(probs+1e-12))
        return data_loss + reg_loss
    def backward(self, y):
        N=y.shape[0]
        C=self.P.shape[1]
        Y = np.zeros((N,C))
        Y[np.arange(N),y]=1
        
        dZ2= (self.P-Y)/N
        dW2=dZ2.T @ self.H
        db2=np.sum(dZ2,axis=0,keepdims=True)
        dH=dZ2 @ self.W2
        dZ1=dH*self.tanh_grad(self.H)

        dW1=dZ1.T @ self.X
        db1=np.sum(dZ1,axis=0,keepdims=True)

        dW2+=self.reg_lambda*self.W2
        dW1+=self.reg_lambda*self.W1

        self.grads={
            "W1":dW1,
            "b1":db1,
            "W2":dW2,
            "b2":db2,
        }

    def step_sgd(self, lr=0.05):
        self.W1 -= lr * self.grads["W1"]
        self.b1 -= lr * self.grads["b1"]
        self.W2 -= lr * self.grads["W2"]
        self.b2 -= lr * self.grads["b2"]

    def step_momentum(self, lr=0.05, beta=0.9):
        if self.velocity is None:
            self.velocity = {
                "W1": np.zeros_like(self.W1),
                "b1": np.zeros_like(self.b1),
                "W2": np.zeros_like(self.W2),
                "b2": np.zeros_like(self.b2),
            }

        for name in ["W1", "b1", "W2", "b2"]:
            self.velocity[name] = beta * self.velocity[name] - lr * self.grads[name]
            if name == "W1": self.W1 += self.velocity[name]
            elif name == "b1": self.b1 += self.velocity[name]
            elif name == "W2": self.W2 += self.velocity[name]
            elif name == "b2": self.b2 += self.velocity[name]

    def step_adam(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        # Explicit None check — safer and clearer
        if self.m is None:
            self.m = {
                "W1": np.zeros_like(self.W1),
                "b1": np.zeros_like(self.b1),
                "W2": np.zeros_like(self.W2),
                "b2": np.zeros_like(self.b2),
            }
            self.v = {
                "W1": np.zeros_like(self.W1),
                "b1": np.zeros_like(self.b1),
                "W2": np.zeros_like(self.W2),
                "b2": np.zeros_like(self.b2),
            }

        self.t += 1

        for name in ["W1", "b1", "W2", "b2"]:
            g = self.grads[name]

            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (g ** 2)

            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)

            update = lr * m_hat / (np.sqrt(v_hat) + eps)

            if name == "W1":
                self.W1 -= update
            elif name == "b1":
                self.b1 -= update
            elif name == "W2":
                self.W2 -= update
            elif name == "b2":
                self.b2 -= update

    def update(self, optimizer="sgd", lr=0.05, momentum_beta=0.9,
               beta1=0.9, beta2=0.999, eps=1e-8):
        if optimizer == "sgd":
            self.step_sgd(lr=lr)
        elif optimizer == "momentum":
            self.step_momentum(lr=lr, beta=momentum_beta)
        elif optimizer == "adam":
            self.step_adam(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def state_dict(self):
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def load_state_dict(self, state):
        self.W1 = state["W1"].copy()
        self.b1 = state["b1"].copy()
        self.W2 = state["W2"].copy()
        self.b2 = state["b2"].copy()