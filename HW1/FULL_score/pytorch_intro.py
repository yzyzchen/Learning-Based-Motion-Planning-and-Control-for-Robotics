import torch

# Type hints.
from typing import List, Tuple
from torch import Tensor


def create_tensor_of_pi(M: int, N: int) -> Tensor:
    """
    Returns a Tensor of shape (M, N) filled entirely with the value 3.14

    Args:
        M, N: Positive integers giving the shape of Tensor to create

    Returns:
        x: A tensor of shape (M, N) filled with the value 3.14
    """
    # --- Your code here
    x = torch.full((M,N),3.14)
    # ---
    return x


def slice_indexing_practice(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Given a two-dimensional tensor x, extract and return several subtensors to
    practice with slice indexing. Each tensor should be created using a single
    slice indexing operation.

    The input tensor should not be modified.

    Args:
        x: Tensor of shape (M, N) -- M rows, N columns with M >= 3 and N >= 5.

    Returns:
        A tuple of:
        - last_row: Tensor of shape (N,) giving the last row of x. It should be
          a one-dimensional tensor.
        - third_col: Tensor of shape (M, 1) giving the third column of x. It
          should be a two-dimensional tensor.
        - first_two_rows_three_cols: Tensor of shape (2, 3) giving the data in
          the first two rows and first three columns of x.
        - even_rows_odd_cols: Two-dimensional tensor containing the elements in
          the even-valued rows and odd-valued columns of x.
    """
    assert x.shape[0] >= 3
    assert x.shape[1] >= 5
    # --- Your code here
    last_row = x[-1,:]
    third_col = x[:,2].unsqueeze(dim = 1)
    first_two_rows_three_cols = x[0:2, 0:3]
    even_rows_odd_cols = x[::2, 1::2]
    # ---

    out = (
        last_row,
        third_col,
        first_two_rows_three_cols,
        even_rows_odd_cols,
    )
    return out


def slice_assignment_practice(x: Tensor) -> Tensor:
    """
    Given a two-dimensional tensor of shape (M, N) with M >= 4, N >= 6, mutate
    its first 4 rows and 6 columns so they are equal to:

    [0 1 2 2 2 2]
    [0 1 2 2 2 2]
    [3 4 3 4 5 5]
    [3 4 3 4 5 5]

    Note: the input tensor shape is not fixed to (4, 6).

    Your implementation must obey the following:
    - You should mutate the tensor x in-place and return it
    - You should only modify the first 4 rows and first 6 columns; all other
      elements should remain unchanged
    - You may only mutate the tensor using slice assignment operations, where
      you assign an integer to a slice of the tensor
    - You must use <= 6 slicing operations to achieve the desired result

    Args:
        x: A tensor of shape (M, N) with M >= 4 and N >= 6

    Returns:
        x
    """
    # --- Your code here
    cube = torch.tensor([
        [0, 1, 2, 2, 2, 2],
        [0, 1, 2, 2, 2, 2],
        [3, 4, 3, 4, 5, 5],
        [3, 4, 3, 4, 5, 5]
    ])
    x[0:4,0:6] = cube
    # ---
    return x


def shuffle_cols(x: Tensor) -> Tensor:
    """
    Re-order the columns of an input tensor as described below.

    Your implementation should construct the output tensor using a single
    integer array indexing operation. The input tensor should not be modified.

    Args:
        x: A tensor of shape (M, N) with N >= 3

    Returns:
        A tensor y of shape (M, 4) where:
        - The first two columns of y are copies of the first column of x
        - The third column of y is the same as the third column of x
        - The fourth column of y is the same as the second column of x
    """
    y = x
    # --- Your code here
    y[:,0] = x[:,0]
    y[:,1] = x[:,0]
    y[:,3] = x[:,1]
    # ---
    return y[:, :4]


def reverse_rows(x: Tensor) -> Tensor:
    """
    Reverse the rows of the input tensor.

    Your implementation should construct the output tensor using a single
    integer array indexing operation. The input tensor should not be modified.

    Args:
        x: A tensor of shape (M, N)

    Returns:
        y: Tensor of shape (M, N) which is the same as x but with the rows
            reversed - the first row of y should be equal to the last row of x,
            the second row of y should be equal to the second to last row of x,
            and so on.
    """
    # --- Your code here
    y = x[torch.arange(x.size(0) - 1, -1, -1),:]
    # ---
    return y


def reshape_practice(x: Tensor) -> Tensor:
    """
    Given an input tensor of shape (24,), return a reshaped tensor y of shape
    (3, 8) such that

    y = [[x[0], x[1], x[2],  x[3],  x[12], x[13], x[14], x[15]],
         [x[4], x[5], x[6],  x[7],  x[16], x[17], x[18], x[19]],
         [x[8], x[9], x[10], x[11], x[20], x[21], x[22], x[23]]]

    You must construct y by performing a sequence of reshaping operations on
    x (view, t, transpose, permute, contiguous, reshape, etc). The input
    tensor should not be modified.

    Args:
        x: A tensor of shape (24,)

    Returns:
        y: A reshaped version of x of shape (3, 8) as described above.
    """
    # --- Your code here
    # Step 1: Reshape the tensor to a shape that groups the elements into rows for manipulation.
    # y = torch.reshape(x, [2, 12])
    # y1 = torch.reshape(y[0,:], [3,4])
    # y2 = torch.reshape(y[1,:], [3,4])
    # y = torch.cat((y1, y2), dim=1)
    y = torch.cat((x[:12].view(3,4), x[12:].view(3,4)), dim = 1)
    # ---
    return y


def batched_matrix_multiply(x: Tensor, y: Tensor) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).

    This implementation should use no explicit Python loops (including
    comprehensions).

    Hint: torch.bmm

    Args:
        x: Tensor of shaper (B, N, M)
        y: Tensor of shape (B, M, P)

    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    # --- Your code here
    z = torch.bmm(x, y)
    # ---
    return z


def compute_scalar_function_and_grad(x: Tensor) -> Tensor:
    """
        Compute the function y = 3*x^2 and compute the gradient using PyTorch autograd

        You only need to return y, if you have successfully computed the gradient using PyTorch autograd the
        gradient will be stored in x.grad
    Args:
        x: Tensor of shape (1).

    Returns:
        y: Tensor of shape (1) as described above. It should have the same
            dtype as the input x.
    """

    # --- Your code here
    x.requires_grad_(True)
    y = 3 * x**2
    y.backward()
    # ---
    return y


def compute_vector_function_and_grad(x: Tensor) -> Tensor:
    """
        Compute the vector function
            y1 = cos(2*x1 + x2)
            y2 = sin(2*x2 - x1)

         and compute the gradient using PyTorch autograd

        You only need to return y, if you have successfully computed the gradient using PyTorch autograd the
        gradient will be stored in x.grad
    Args:
        x: Tensor of shape (2).

    Returns:
        y: Tensor of shape (2) as described above. It should have the same
            dtype as the input x.
    """
    # --- Your code here
    x.requires_grad_()
    y1 = torch.cos(2*x[0] + x[1])
    y2 = torch.sin(2*x[1] - x[0])
    y = torch.stack([y1, y2])
    y.sum().backward()
    # ---
    return y


def compute_scalar_function_and_partial_grad(x: Tensor, y: Tensor) -> Tensor:
    """
        Compute the vector function
            z = x^0.5 * y

         and compute the gradient using PyTorch autograd ONLY with respect to x - we do not want to compute the
         gradient with respect to y

        You only need to return y, if you have successfuly computed the gradient using PyTorch autograd the
        gradient will be stored in x.grad BUT the gradient stored in y.grad should be None
    Args:
        x: Tensor of shape (1).
        y: Tensor of shape (1).

    Returns:
        z: Tensor of shape (1) as described above. It should have the same
            dtype as the input x.
    """

    # --- Your code here
    x.requires_grad_()
    y.detach()
    z = x**0.5 * y
    z.backward()
    # ---
    return z


def compute_forward_kinematics(thetas: torch.Tensor) -> torch.Tensor:
    """
    Compute the forward kinematics of the robot configuration given by theta
    Args:
        thetas: Pytorch Tensor of shape (2,) containing the robot joints
    Returns:
        x: Pytorch Tensor of shape (2,) containing the end-effector position

    """
    L1 = 2
    L2 = 1

    # --- Your code here
    thetas.requires_grad_()
    x1 = L1 * torch.cos(thetas[0]) + L2 * torch.cos(thetas[0] + thetas[1])
    x2 = L1 * torch.sin(thetas[0]) + L2 * torch.sin(thetas[0] + thetas[1])
    x = torch.stack([x1,x2])
    # ---
    return x


def compute_jacobian(thetas: torch.Tensor) -> torch.Tensor:
    """
    Compute the manipulator Jacobian
    Args:
        thetas: Pytorch Tensor of shape (2,) containing the robot joints
    Returns:
        J: Pytorch Tensor of shape (2,2) containing the end-effector position

    """
    # --- Your code here
    # Ensure gradients are tracked for thetas
    thetas.requires_grad_(True)

    # Compute the forward kinematics
    x = compute_forward_kinematics(thetas)  # x is of shape (2,)

    # Initialize an empty list to hold rows of the Jacobian
    J = []

    # Compute gradients for each component of x (x1, x2)
    for i in range(x.shape[0]):
        # Compute gradient of x[i] with respect to thetas
        grad = torch.autograd.grad(outputs=x[i], inputs=thetas, create_graph=True, retain_graph=True)[0]
        J.append(grad)  # Append the gradient (a row of the Jacobian)

    # Stack the rows to form the Jacobian matrix
    J = torch.stack(J)
    # ---
    return J
