import numpy as np

def points(time, width, height):
    # original set
    set1 = np.array([
        [0.1 * width, height * 0.9],  # bottom left
        [0.45 * width, 0.55 * height],  # left midpoint
        [0.55 * width, 0.55 * height],  # right midpoint
        [0.9 * width, height * 0.9]  # bottom right
    ], np.int32)

    #for beginning
    set2 = np.array([
        [0.2 * width, height * 0.9],  # bottom left
        [0.45 * width, 0.58 * height],  # left midpoint
        [0.55 * width, 0.58 * height],  # right midpoint
        [0.8 * width, height * 0.9]  # bottom right
    ], np.int32)

    #for middle
    set3 = np.array([
        [0.2 * width, height * 0.9],  # bottom left
        [0.45 * width, 0.58 * height],  # left midpoint
        [0.58 * width, 0.58 * height],  # right midpoint
        [0.8 * width, height * 0.9]  # bottom right
    ], np.int32)

    #for middle end
    set4 = np.array([
        [0.1 * width, height * 0.9],  # bottom left
        [0.45 * width, 0.58 * height],  # left midpoint
        [0.5 * width, 0.58 * height],  # right midpoint
        [0.8 * width, height * 0.9]  # bottom right
    ], np.int32)

    #for end
    set5 = np.array([
        [0.2 * width, height],  # bottom left
        [0.48 * width, 0.7 * height],  # left midpoint
        [0.5 * width, 0.7 * height],  # right midpoint
        [0.8 * width, height ]  # bottom right
    ], np.int32)

    print(int(time))
    if time < 61 or 340<time<1000: 
        return set2
    # elif 110 < time < 140:
    #     return set3
    elif 200 < time < 280:
        return set4
    elif 290 < time < 320:
        return set5
    else:
        return set1
