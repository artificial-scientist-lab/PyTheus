## Experimental files

- **analytic_gradient_theseus626.py**: Shows how to use PyTorch's analytic gradients to optimize the (6,2,6)-GHZ state, using the [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) optimizer. Seems to be significantly faster than the numeric gradients via BFGS, used in the main code.
