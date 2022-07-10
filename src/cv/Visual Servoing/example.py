from visual_servo import VisualServo
import numpy as np
import timeit

def printing_messages(status, origin_pos, sample_joint, start, stop, test_id):
    print("############################# start of test" + str(test_id) + " #############################")
    if status[0] == 1:
        print("there is z0 collision")
        print("origin_pos is: ", origin_pos, "collision at: ", status[1])
    elif status[0] == 2:
        print("there is self collision")
        print("joint values are: ", sample_joint*180/np.pi, "collision at: ", status[1])
    elif status[0] == 3:
        print("there are both self and z0 collisions")
        print("origin_pos is: ", origin_pos, "joint values are: ", sample_joint * 180 / np.pi, "collision at: ", status[1])
    else:
        print("there is no collision")
    #print("jacobian is: ", VS.Jacobi_camera)
    print("joint velocity is: ", joint_vel)
    print("runtime is: ", stop - start)
    print("############################# end of test" + str(test_id) + " #############################")
if __name__ == '__main__':
    VS = VisualServo()
    test_failed_ID = []
    #test 0: random
    test_id = 0
    sample_joint = np.array([0.1,np.pi/1.5,0.1,0.1,0.1, 0.2])
    sample_goal = np.array([[0],[0],[0],[0],[0],[0]]) # zero vector for orientation
    sample_ball_coord = np.array([[-0.05],[-0.1],[-0.08],[0],[0],[0]]) # zero vector for orientation
    sample_joint_vel = np.array([[0.0], [0.01], [0.1], [0.05], [0.05], [0.1]])
    ball_vel = np.array([[0.02], [0.02], [0.02]])
    ball_detected = True
    predicted_pos = np.array([[-100], [-100], [-100]])
    start = timeit.default_timer()
    status, origin_pos, joint_vel = VS.visual_servoing(sample_joint, sample_joint_vel, sample_goal, sample_ball_coord, ball_vel, ball_detected, predicted_pos)
    stop = timeit.default_timer()
    printing_messages(status, origin_pos, sample_joint, start, stop, test_id)

    #test 1: testing link 4 + 5 collision
    test_id += 1
    sample_joint = np.array([0, 0, 119, 50, 6, 12])*np.pi/180
    sample_goal = np.array([[0], [0], [0], [0], [0], [0]])  # zero vector for orientation
    sample_ball_coord = np.array([[-0.3], [0.2], [0.4], [0], [0], [0]])  # zero vector for orientation
    sample_joint_vel = np.array([[0.0], [0.01], [0.1], [0.05], [0.05], [0.1]])
    start = timeit.default_timer()
    status, origin_pos, joint_vel = VS.visual_servoing(sample_joint, sample_joint_vel, sample_goal, sample_ball_coord,ball_vel, ball_detected, predicted_pos)
    stop = timeit.default_timer()
    printing_messages(status, origin_pos, sample_joint, start, stop, test_id)
    if status[0] == 2:
        print("test passed, status is: ", status)
    else:
        print("test failed, status is: ", status)
        test_failed_ID.append(test_id)

    #test 2: test link 4 + 5 collision with negative angles
    test_id += 1
    sample_joint = np.array([0, 0, -119, -50, 6, 12]) * np.pi / 180
    sample_goal = np.array([[0], [0], [0], [0], [0], [0]])  # zero vector for orientation
    sample_ball_coord = np.array([[0.3], [0.2], [0.4], [0], [0], [0]])  # zero vector for orientation
    sample_joint_vel = np.array([[0.0], [0.01], [0.1], [0.05], [0.05], [0.1]])
    start = timeit.default_timer()
    status, origin_pos, joint_vel = VS.visual_servoing(sample_joint, sample_joint_vel, sample_goal, sample_ball_coord, ball_vel, ball_detected, predicted_pos)
    stop = timeit.default_timer()
    printing_messages(status, origin_pos, sample_joint, start, stop, test_id)
    if status[0] == 2:
        print("test passed, status is: ", status)
    else:
        print("test failed, status is: ", status)
        test_failed_ID.append(test_id)

    #test 3: test link 1 collision with negative angles
    test_id += 1
    sample_joint = np.array([170, 0, 0, -50, 6, 12]) * np.pi / 180
    sample_goal = np.array([[0], [0], [0], [0], [0], [0]])  # zero vector for orientation
    sample_ball_coord = np.array([[0.3], [0.2], [0.4], [0], [0], [0]])  # zero vector for orientation
    sample_joint_vel = np.array([[0.0], [0.01], [0.1], [0.05], [0.05], [0.1]])
    start = timeit.default_timer()
    status, origin_pos, joint_vel = VS.visual_servoing(sample_joint, sample_joint_vel, sample_goal, sample_ball_coord, ball_vel, ball_detected, predicted_pos)
    stop = timeit.default_timer()
    printing_messages(status, origin_pos, sample_joint, start, stop, test_id)
    if status[0] == 2:
        print("test passed, status is: ", status)
    else:
        print("test failed, status is: ", status)
        test_failed_ID.append(test_id)

    #test 4: test link 3,4,5,6 collision with z0
    test_id += 1
    sample_joint = np.array([0, 90, 80, 50, 6, 12]) * np.pi / 180
    sample_goal = np.array([[0], [0], [0], [0], [0], [0]])  # zero vector for orientation
    sample_ball_coord = np.array([[0.3], [0.2], [0.4], [0], [0], [0]])  # zero vector for orientation
    sample_joint_vel = np.array([[0.0], [0.01], [0.1], [0.05], [0.05], [0.1]])
    start = timeit.default_timer()
    status, origin_pos, joint_vel = VS.visual_servoing(sample_joint, sample_joint_vel, sample_goal, sample_ball_coord, ball_vel, ball_detected, predicted_pos)
    stop = timeit.default_timer()
    printing_messages(status, origin_pos, sample_joint, start, stop, test_id)
    if status[0] == 1 and status[1] == [3, 4, 5, 6]:
        print("test passed, status is: ", status)
    else:
        print("test failed, status is: ", status)
        test_failed_ID.append(test_id)

    #test 5: test link 3 and 4 collision with z0, as if link 4 collides with z0, link 5 will as well
    test_id += 1
    sample_joint = np.array([0, 110, 0, -100, 6, 12]) * np.pi / 180
    sample_goal = np.array([[0], [0], [0], [0], [0], [0]])  # zero vector for orientation
    sample_ball_coord = np.array([[0.3], [0.2], [0.4], [0], [0], [0]])  # zero vector for orientation
    sample_joint_vel = np.array([[0.0], [0.01], [0.1], [0.05], [0.05], [0.1]])
    start = timeit.default_timer()
    status, origin_pos, joint_vel = VS.visual_servoing(sample_joint, sample_joint_vel, sample_goal, sample_ball_coord, ball_vel, ball_detected, predicted_pos)
    stop = timeit.default_timer()
    printing_messages(status, origin_pos, sample_joint, start, stop, test_id)
    if status[0] == 1 and status[1] == [3, 4]:
        print("test passed, status is: ", status)
    else:
        print("test failed, status is: ", status)
        test_failed_ID.append(test_id)
    print("origin_pos is: ", origin_pos, "joint values are: ", sample_joint * 180 / np.pi, "collision at: ", status[1])

    # test 6: testing orientation for when the ball is too high and ball is in the frame
    test_id += 1
    sample_ball_coord = np.array([[0.3], [0.2], [VS.max_horizontal_look_height + VS.max_height_padding/2], [0], [0], [0]])  # zero vector for orientation
    sample_joint = np.array([0, 0, 0, 0, 0, 90]) * np.pi / 180
    sample_joint_vel = np.array([[0.0], [0.01], [0.1], [0.05], [0.05], [0.1]])
    ball_vel = np.array([[0.02], [0.02], [0.02]])
    ball_detected = True


    sample_goal = np.array([[0], [0], [0], [0], [0], [0]])  # zero vector for orientation

    start = timeit.default_timer()
    status, origin_pos, joint_vel = VS.visual_servoing(sample_joint, sample_joint_vel, sample_goal, sample_ball_coord,
                                                       ball_vel, ball_detected, predicted_pos)
    stop = timeit.default_timer()
    printing_messages(status, origin_pos, sample_joint, start, stop, test_id)
    VS_ori = VS.vector2rpy((sample_ball_coord[0:3]) * (-1))
    if all(VS.desired_orientation == VS_ori):
        print("test passed, desired_orientation is: ", VS.desired_orientation)
    else:
        print("test failed, desired_orientation is: ", VS.desired_orientation, "actual one got is: ", VS_ori)
        test_failed_ID.append(test_id)


    # test 7: testing orientation for when the ball is too low and ball is in the frame
    test_id += 1
    sample_ball_coord = np.array(
        [[0.3], [0.2], [0 + VS.min_height_padding / 2], [0], [0],
        [0]])  # zero vector for orientation
    sample_joint = np.array([0, 0, 0, 0, 0, 90]) * np.pi / 180
    sample_joint_vel = np.array([[0.0], [0.01], [0.1], [0.05], [0.05], [0.1]])
    ball_vel = np.array([[0.02], [0.02], [0.02]])
    ball_detected = True

    sample_goal = np.array([[0], [0], [0], [0], [0], [0]])  # zero vector for orientation

    start = timeit.default_timer()
    status, origin_pos, joint_vel = VS.visual_servoing(sample_joint, sample_joint_vel, sample_goal,
                                                        sample_ball_coord,
                                                        ball_vel, ball_detected, predicted_pos)
    stop = timeit.default_timer()
    printing_messages(status, origin_pos, sample_joint, start, stop, test_id)
    VS_ori = VS.vector2rpy((sample_ball_coord[0:3] - np.array([[0],[0],[VS.min_height_padding]])) * (-1))
    if all(VS.desired_orientation == VS_ori):
            print("test passed, desired_orientation is: ", VS.desired_orientation)
    else:
        print("test failed, desired_orientation is: ", VS.desired_orientation, "actual one got is: ", VS_ori)
        test_failed_ID.append(test_id)

    # test 8: testing orientation for when the ball is too low and ball is NOT in the frame
    test_id += 1
    sample_ball_coord = np.array(
        [[0.3], [0.2], [0 + VS.min_height_padding / 2], [0], [0],
        [0]])  # zero vector for orientation
    sample_joint = np.array([0, 0, 0, 0, 0, 90]) * np.pi / 180
    sample_joint_vel = np.array([[0.0], [0.01], [0.1], [0.05], [0.05], [0.1]])
    ball_vel = np.array([[0.02], [0.02], [-0.02]])
    ball_detected = False
    predicted_pos = np.array([[0.4], [0.5], [0]])

    sample_goal = np.array([[0], [0], [0], [0], [0], [0]])  # zero vector for orientation

    start = timeit.default_timer()
    status, origin_pos, joint_vel = VS.visual_servoing(sample_joint, sample_joint_vel, sample_goal,
                                                        sample_ball_coord,
                                                        ball_vel, ball_detected, predicted_pos)
    stop = timeit.default_timer()
    printing_messages(status, origin_pos, sample_joint, start, stop, test_id)
    VS_ori = VS.vector2rpy((predicted_pos[0:3] - np.array([[0],[0],[VS.min_height_padding]])) * (-1))
    if all(VS.desired_orientation == VS_ori):
            print("test passed, desired_orientation is: ", VS.desired_orientation)
    else:
        print("test failed, desired_orientation is: ", VS.desired_orientation, "actual one got is: ", VS_ori)
        test_failed_ID.append(test_id)

    # test 9: testing orientation for when it is normal
    test_id += 1
    sample_ball_coord = np.array(
        [[0.3], [0.2], [0.2], [0], [0],
        [0]])  # zero vector for orientation
    sample_joint = np.array([0, 0, 0, 0, 0, 90]) * np.pi / 180
    sample_joint_vel = np.array([[0.0], [0.01], [0.1], [0.05], [0.05], [0.1]])
    ball_vel = np.array([[0.02], [0.02], [-0.02]])
    ball_detected = True
    predicted_pos = np.array([[0.4], [0.5], [0]])

    sample_goal = np.array([[0], [0], [0], [0], [0], [0]])  # zero vector for orientation

    start = timeit.default_timer()
    status, origin_pos, joint_vel = VS.visual_servoing(sample_joint, sample_joint_vel, sample_goal,
                                                        sample_ball_coord,
                                                        ball_vel, ball_detected, predicted_pos)
    stop = timeit.default_timer()
    printing_messages(status, origin_pos, sample_joint, start, stop, test_id)
    VS_ori = np.array([[0],[0],[0]])
    if all(VS.desired_orientation == VS_ori):
            print("test passed, desired_orientation is: ", VS.desired_orientation)
    else:
        print("test failed, desired_orientation is: ", VS.desired_orientation, "actual one got is: ", VS_ori)
        test_failed_ID.append(test_id)






    if (len(test_failed_ID) == 0):
        print("all tests passed!!")
    else:
        print("some tests failed, all failed tests are: ", test_failed_ID)
