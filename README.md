
# OpenVLA Gripper Predict is 46% off:
I took the first episode from the fractal20220817 data and ran predict with OpenVLA and got results for the gripper that are 46% off.
This is critical since if the gripper opens or closes in the wrong moment the whole movement will fail.

Please see attached documents:

1. [Images (gif) used for running the prediction](https://github.com/lovecode1/OpenVLA-Gripper-Accuracy-Issue/blob/main/fractal20220817_data_Episode_1.gif)

2. Prompt: "In: What action should the robot take to pick rxbar chocolate from bottom drawer and place on counter?\n Out:"

3. [Episode data action list](https://github.com/lovecode1/OpenVLA-Gripper-Accuracy-Issue/blob/main/fractal20220817_data_original_actions.csv)

4. [Predicted actions](https://github.com/lovecode1/OpenVLA-Gripper-Accuracy-Issue/blob/main/fractal20220817_data_predicted_actions.csv)
5. [Python script to run the predict](https://github.com/lovecode1/OpenVLA-Gripper-Accuracy-Issue/blob/main/run_predict.py)
