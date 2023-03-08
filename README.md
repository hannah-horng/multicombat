# multicombat

This is the repository for **MultiComBat harmonization for multiple batch variables**, currently under review at MICCAI 2023. MultiComBat harmonization generalizes the ComBat harmonization framework to multiple batch variables (e.g. Vendor, location) and completes estimation for the corresponding estimates in a single iteration. 

## Updates are still ongoing!
- These methods are still a work in progress, so please be advised to check regularly for updates!
- Some ongoing issues that we are aware of that we recommend caution with are:
    - **Unstable variance estimation**: MultiComBat models the scale effects of multiple batch variables as a product, resulting in significant variance shrinkage. Users should be advised to use "mean_only=True" to disable scale estimation until we implement an effective constraint.
    - **Code readability**: We are working on cleaning the code to make it more straightforward.

## Repository information
Functions for executing MultiComBat are found in **multicombat.py**.

An example with simulated data can be found in **multicombat_demo.py**. This example obtains MultiComBat estimates from under-sampled training dataset, then applies these estimates to a much larger testing dataset. Example outputs can be found in **demo_output**.
