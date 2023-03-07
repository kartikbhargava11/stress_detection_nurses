# 1. Project Description

You’re a data scientist in a company that has developed a wearable watch with multiple sensors. The company is trying to develop a new feature that involves alerting a user when it detects that they’re getting stressed and has asked you to investigate whether this is possible with the sensors that are currently available on the device. For this, the company has run an experiment with nurses in a hospital, in which the nurses wore the watch for periods of time and reported their corresponding levels of stress. You have been given access to the Nurse stress data set and your manager has asked you to investigate whether this new feature can be implemented using the available sensors. The company is particularly worried about false negatives, as they are trying to
convince the hospital to sign a contract with them and this is conditional on having high recall. The report describing the data set and containing instructions on how to access it is available here [1].
You need to show that you understand the signals that are being recorded, how to properly process them
and find out whether and how they can be used to detect and/or forecast stress so that you can make a
recommendation to the company and present your results concisely, paying special attention to how the
algorithm is making the decision and which of the signals can and cannot be used for this task.
Read the following tasks in detail and make sure you understand the project.
# 2. Tasks
## 2.1 Assignment 1
1. Load and explore the data set, leaving a subset of data separate from the exploration to avoid overfitting.
You cannot use the code from the GitHub repository for this data set and present it as your
own.
2. Clean and preprocess the data set.
3. From your exploration only, find out which signals might be better candidates for predicting stress. You
can run some early modelling tests, but this is not strictly necessary for this assignment. During your
demo, we will check that you understand the different signals and have a plan for how to analyse the data
for Assignment 2 that is reasonable and feasible.
## 2.2 Assignment 2
1. The modelling is up to you, as long as it follows good practice in data science (i.e., using cross-validation
and train/test splits properly and as appropriate for the problem and the chosen modelling approach).
You are encouraged to check out the available literature and find out what has worked in the past, but
you need to present something new/different as part of your project. You cannot use the code from
the GitHub repository for this data set and present it as your own.
2. While you’re modelling, reflect on the project description given above and make sure you pay special
attention to the requirements set out by your manager. You may use all signals or a subset of them only,
and you can choose to only detect stress or trying to forecast it in advance. This is all up to you.
3. For your report, make sure that all your decisions are justified and that you present your findings clearly
and concisely following the template given to you on Moodle. As a data scientist, you must reflect on
your results. Talk about your findings and what they indicate. Finally, you must answer the question: are
these signals useful for predicting stress? Can the company guarantee good performance to the hospital
and go ahead with this contract? Do you have any other insights from the data that can help the company
in the future (e.g., for future features that can be commercialised)?
1
# References
[1] S. Hosseini, R. Gottumukkala, S. Katragadda, R. T. Bhupatiraju, Z. Ashkar, C. W. Borst, and K. Cochran.
A multimodal sensor dataset for continuous stress detection of nurses in a hospital. Scientific Data, 9(1):1–13,
2022.