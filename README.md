# Darkside20k
Hello to all that would like to see this code!


Here are the steps to follow in order to properly use it:
- Make sure you have all the imports needed
- Go and graph histogram of datapoints to find left and right bounds of singlet templates
    - Should be just left and right of the first peak in histogram
    - Adjust "singlet_left_bound" and "singlet_right_bound"
        - go and adjust these values on all files now before you forget

- To see scatterplot go to bottom of scatterplotanalysis for useful functions to sift through
- Have to manually make the boarders for the rectangles to remove datapoints and count up each type of noise
- Use the tuple values for rectangles as described when you want to make more rectangles:
    - (left_limit, right_limit, lower_limit, upper_limit)

- Future statistical modeling could be used to show how each type of noise occures
    - How do neighboring SPADs interact??
    - Use our proportions to compare single vs double vs ... DiCT and compare it to models

- Recharge Exponential was used to model the AP recharge rate
    - Useful interactive map to help guide future modeling
