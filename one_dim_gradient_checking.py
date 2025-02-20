import numpy as np

def forward_propagation(x, theta):
    J = np.dot(theta, x)
    return J

def backward_propagation(x, theta):
    dtheta = x
    return dtheta

def gradient_checking(x, theta, epsilon = 1e-7):
    theta_plus_epsilon = theta + epsilon
    theta_minus_epsilon = theta - epsilon
    J_plus_epsilon = x * theta_plus_epsilon
    J_minus_epsilon = x * theta_minus_epsilon
    gradient = (J_plus_epsilon - J_minus_epsilon) / (2*epsilon)
    return gradient


def main():
    x, theta = 2, 4
    J = forward_propagation(x, theta)
    print("J: ", J)
    dtheta = backward_propagation(x, theta)
    print("dtheta: ", dtheta)
    gradient = gradient_checking(x, theta)
    numerator = np.linalg.norm(gradient - dtheta)
    denominator = np.linalg.norm(gradient) + np.linalg.norm(dtheta)
    check = numerator / denominator

    print("difference: ", check)

    if check <= 1e-7:
        print("gradient is correct")
    else:
        print("gradient is incorrect")

if __name__ == "__main__":
    main()