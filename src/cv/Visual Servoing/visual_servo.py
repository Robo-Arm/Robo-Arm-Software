import numpy as np
from math import cos, sin, acos, asin

class VisualServo:
    def __init__(self):
        self.Jacobi_camera = None
        self.J3 = None
        self.J4 = None
        self.J5 = None
        self.J6 = None
        self.T03 = None
        self.T3 = None
        self.T4 = None
        self.T5 = None
        self.T6 = None
        self.clamp = 1 # TO BE TUNED: to prevent the error of goal and ball pos to be too small
        self.speed_reduce = 1 # TO BE TUNED: constant to reduce the speed if too big so that the motion will be smooth
        #self.self_collision_constraint = np.array([[140, 0, 0, 0, 0, 0],[0, 120, 0,0,0,0], [0,0,170,0,0,0],[0,0,0,170,0,0],[0,0,120, 40, 0, 0],[0,0,120, 40, 0, 0]])
        self.self_collision_constraint = np.array([[140, 0, 0, 0, 0, 0],[0, 120, 0,0,0,0], [0,0,170,0,0,0],[0,0,0,170,0,0],[0,0,120, 40, 0, 0],[0,0,120, 40, 0, 0]])
        self.ground_collision_constraint = None
        self.collision_radius = [0.03, 0.03, 0.06, 0.034, 0.06]# in meters
        self.padding = [0.03, 0.005] #2cm of padding for the velocity to decrease
        self.angle_padding = 10
        self.list_of_functions = [self.get_J3, self.get_J4, self.get_J5, self.get_J6]
        self.end_eff_length = 0.08
        self.a = [0,-0.3,-0.18, 0,0,0]
        self.d = [.2,0,0,.0935,.08,self.end_eff_length/2]
        self.max_horizontal_look_height = abs(self.d[0]) + abs(self.a[1]) + abs(self.a[1]) + abs(self.d[4])
        self.max_height_padding = 0.05
        self.min_height_padding = 0.096
        self.desired_orientation = None
    def degree2radians(self, q_1, q_2, q_3, q_4, q_5, q_6):
        q_1 = q_1 * 180 / np.pi
        q_2 = q_2 * 180 / np.pi
        q_3 = q_3 * 180 / np.pi
        q_4 = q_4 * 180 / np.pi
        q_5 = q_5 * 180 / np.pi
        q_6 = q_6 * 180 / np.pi
        return q_1, q_2, q_3, q_4, q_5, q_6
    def get_J3(self, q_1, q_2, q_3, q_4, q_5, q_6):
        #q_1, q_2, q_3, q_4, q_5, q_6 = self.degree2radians(q_1, q_2, q_3, q_4, q_5, q_6)
        t2 = cos(q_1)
        t3 = sin(q_1)
        t4 = q_1 + q_2
        t5 = q_2 + q_3
        t10 = -q_1
        t11 = -q_2
        t6 = cos(t4)
        t7 = q_3 + t4
        t8 = sin(t4)
        t9 = sin(t5)
        t14 = -t2
        t15 = q_1 + t11
        t17 = t5 + t10
        t12 = cos(t7)
        t13 = sin(t7)
        t16 = cos(t15)
        t18 = sin(t15)
        t19 = cos(t17)
        t20 = sin(t17)
        t21 = t9* (1.59e+2/ 1.0e+3)
        t23 = t6* 1.5e-1
        t24 = t8* 1.5e-1
        t22 = -t21
        t25 = -t23
        t26 = -t24
        t27 = t12* 7.95e-2
        t28 = t13* 7.95e-2
        t31 = t16* 1.5e-1
        t32 = t18* 1.5e-1
        t33 = t19* 7.95e-2
        t34 = t20* 7.95e-2
        t29 = -t27
        t30 = -t28
        t35 = -t33
        mt1 = [t2/ 2.0e+2 + t25 + t29 + t31 + t33, t3/ 2.0e+2 + t26 + t30 + t32 - t34, 0.0, t3, t14,
               6.123233995736766e-17, t25 + t29 - t31 + t35, t26 + t30 - t32 + t34, t22 - sin(q_2)* (3.0/ 1.0e+1),
               t3, t14, 6.123233995736766e-17, t29 + t35, t30 + t34, t22, t3, t14]
        mt2 = [6.123233995736766e-17, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0]
        for x in mt2:
            mt1.append(x)
        self.J3 = np.reshape(mt1, (6,6), order="F")
        return self.J3
    def get_J4(self, q_1, q_2, q_3, q_4, q_5, q_6):
        #q_1, q_2, q_3, q_4, q_5, q_6 = self.degree2radians(q_1, q_2, q_3, q_4, q_5, q_6)
        t2 = cos(q_1)
        t3 = sin(q_1)
        t4 = q_1 + q_2
        t5 = q_2 + q_3
        t11 = -q_1
        t12 = -q_2
        t6 = cos(t4)
        t7 = q_3 + t4
        t8 = q_4 + t5
        t9 = sin(t4)
        t10 = sin(t5)
        t17 = -t2
        t20 = q_1 + t12
        t22 = t5 + t11
        t13 = cos(t7)
        t14 = q_4 + t7
        t15 = sin(t7)
        t16 = sin(t8)
        t21 = cos(t20)
        t23 = sin(t20)
        t24 = cos(t22)
        t25 = t8 + t11
        t26 = sin(t22)
        t29 = t10* (9.0/ 5.0e+1)
        t33 = t6* 1.5e-1
        t34 = t9* 1.5e-1
        t18 = cos(t14)
        t19 = sin(t14)
        t27 = cos(t25)
        t28 = sin(t25)
        t30 = -t29
        t31 = t16/ 1.0e+2
        t35 = -t33
        t36 = -t34
        t37 = t13* 9.000000000000001e-2
        t38 = t15* 9.000000000000001e-2
        t45 = t21* 1.5e-1
        t46 = t23* 1.5e-1
        t49 = t24* 8.999999999999999e-2
        t51 = t26* 8.999999999999999e-2
        t32 = -t31
        t39 = t18* 5.0e-3
        t40 = t19* 5.0e-3
        t41 = -t37
        t42 = -t38
        t47 = t27* 5.0e-3
        t48 = t28* 5.0e-3
        t52 = -t49
        t43 = -t39
        t44 = -t40
        t50 = -t47
        mtt = []
        mt1 = [t2* 9.35e-2 + t35 + t41 + t43 + t45 + t47 + t49, t3* 9.35e-2 + t36 + t42 + t44 + t46 - t48 - t51,
               0.0, t3, t17]
        mtt.append(mt1)
        mtt.append([6.123233995736766e-17, t35 + t41 + t43 - t45 + t50 + t52, t36 + t42 + t44 - t46 + t48 + t51,
               t30 + t32 - sin(q_2)* (3.0/ 1.0e+1), t3, t17, 6.123233995736766e-17, t41 + t43 + t50 + t52,
               t42 + t44 + t48 + t51, t30 + t32, t3, t17, 6.123233995736766e-17])
        mtt.append([t43 + t50, t44 + t48, t32, t3* 6.123233995736766e-17 - t19* 5.0e-1 - t28* 5.0e-1])
        mtt.append([t2* (-6.123233995736766e-17) + t18* 5.0e-1 - t27* 5.0e-1])
        mtt.append([cos(t8) + 3.749399456654644e-33, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        print("mt1 is:", mt1 )
        mt1 = []
        for x in mtt:
            #print("x is: ", x)
            for y in x:
                #print("y is: ", y)
                mt1.append(y)
        self.J4 = np.reshape(mt1, (6, 6), order="F")
        return self.J4

    def get_J5(self, q_1, q_2, q_3, q_4, q_5, q_6):
        #q_1, q_2, q_3, q_4, q_5, q_6 = self.degree2radians(q_1, q_2, q_3, q_4, q_5, q_6)
        t2 = cos(q_1)
        t3 = sin(q_1)
        t4 = q_1 + q_2
        t5 = q_2 + q_3
        t6 = q_1 + q_5
        t14 = -q_1
        t15 = -q_2
        t16 = -q_5
        t7 = cos(t4)
        t8 = cos(t6)
        t9 = q_3 + t4
        t10 = q_4 + t5
        t11 = sin(t4)
        t12 = sin(t5)
        t13 = sin(t6)
        t23 = -t2
        t28 = q_1 + t15
        t29 = q_1 + t16
        t33 = t5 + t14
        t17 = cos(t9)
        t18 = cos(t10)
        t19 = q_4 + t9
        t20 = q_5 + t10
        t21 = sin(t9)
        t22 = sin(t10)
        t30 = cos(t28)
        t31 = cos(t29)
        t34 = sin(t28)
        t35 = sin(t29)
        t37 = cos(t33)
        t38 = t10 + t14
        t39 = t10 + t16
        t40 = sin(t33)
        t46 = t8/ 4.0e+2
        t47 = t12* (9.0/ 5.0e+1)
        t48 = t13/ 4.0e+2
        t54 = -t6 + t10
        t59 = t7* 1.5e-1
        t60 = t11* 1.5e-1
        t24 = cos(t19)
        t25 = q_5 + t19
        t26 = sin(t19)
        t27 = sin(t20)
        t41 = cos(t38)
        t42 = t16 + t19
        t43 = t14 + t20
        t44 = sin(t38)
        t45 = sin(t39)
        t53 = -t47
        t55 = t31/ 4.0e+2
        t56 = t35/ 4.0e+2
        t57 = cos(t54)
        t58 = sin(t54)
        t61 = -t59
        t62 = -t60
        t63 = t17* 9.000000000000001e-2
        t64 = t21* 9.000000000000001e-2
        t68 = t22* 8.0e-2
        t70 = t30* 1.5e-1
        t71 = t34* 1.5e-1
        t73 = t37* 8.999999999999999e-2
        t74 = t40* 8.999999999999999e-2
        t32 = sin(t25)
        t36 = cos(t25)
        t49 = cos(t42)
        t50 = cos(t43)
        t51 = sin(t42)
        t52 = sin(t43)
        t65 = t27* 2.5e-3
        t66 = -t63
        t67 = -t64
        t69 = -t68
        t72 = t45* 2.5e-3
        t76 = -t73
        t77 = t26* 4.0e-2
        t80 = t24* 4.0e-2
        t90 = t41* 4.0e-2
        t91 = t44* 4.0e-2
        t93 = t57* 1.25e-3
        t94 = t58* 1.25e-3
        t75 = -t72
        t78 = t36* 1.25e-3
        t79 = t32* 1.25e-3
        t81 = -t77
        t82 = -t80
        t83 = t49* 1.25e-3
        t84 = t50* 1.25e-3
        t85 = t51* 1.25e-3
        t86 = t52* 1.25e-3
        t92 = -t90
        t95 = -t93
        t96 = -t94
        t87 = -t83
        t88 = -t85
        t89 = -t86
        et1 = t3* 3.749399456654644e-33 + t13/ 2.0
        et2 = t26* (-3.061616997868383e-17)
        et3 = t32* 2.5e-1 + t35/ 2.0
        et4 = t44* (-3.061616997868383e-17)
        et5 = t51* (-2.5e-1)
        et6 = t52* 2.5e-1
        et7 = t58* (-2.5e-1)
        et8 = t2* (-3.749399456654644e-33) - t8/ 2.0
        et9 = t24* 3.061616997868383e-17 - t31/ 2.0
        et10 = t36* (-2.5e-1)
        et11 = t41* (-3.061616997868383e-17)
        et12 = t49* 2.5e-1
        et13 = t50* 2.5e-1
        et14 = t57* (-2.5e-1)
        et15 = t18* 6.123233995736766e-17 + cos(q_5)* 6.123233995736766e-17 - cos(t20)* 5.0e-1
        et16 = cos(t39)* 5.0e-1
        et17 = 2.295845021658468e-49
        mtt = []
        mt1 = [t2* 9.35e-2 + t46 + t55 + t61 + t66 + t70 + t73 + t78 + t82 - t84 + t87 + t90 + t93]
        mtt.append(mt1)
        mtt.append([t3* 9.35e-2 + t48 + t56 + t62 + t67 + t71 - t74 + t79 + t81 + t86 + t88 - t91 + t96, 0.0, t3, t23])
        mtt.append([6.123233995736766e-17, t61 + t66 - t70 + t76 + t78 + t82 + t84 + t87 + t92 + t95,
               t62 + t67 - t71 + t74 + t79 + t81 + t88 + t89 + t91 + t94,
               t53 + t65 + t69 + t75 - sin(q_2)* (3.0/ 1.0e+1), t3, t23, 6.123233995736766e-17,
               t66 + t76 + t78 + t82 + t84 + t87 + t92 + t95, t67 + t74 + t79 + t81 + t88 + t89 + t91 + t94,
               t53 + t65 + t69 + t75, t3, t23])
        mtt.append([6.123233995736766e-17, t78 + t82 + t84 + t87 + t92 + t95, t79 + t81 + t88 + t89 + t91 + t94,
               t65 + t69 + t75])
        mtt.append([t3* 6.123233995736766e-17 - t26* 5.0e-1 - t44* 5.0e-1])
        mtt.append([t2* (-6.123233995736766e-17) + t24* 5.0e-1 - t41* 5.0e-1])
        mtt.append([t18 + 3.749399456654644e-33, t46 - t55 + t78 + t83 + t84 + t93, t48 - t56 + t79 + t85 + t89 + t96,
               t65 + t72 - sin(q_5)* 3.061616997868383e-19, et1 + et2 + et3 + et4 + et5 + et6 + et7,
               et8 + et9 + et10 + et11 + et12 + et13 + et14, et15 + et16 + et17, 0.0])
        mtt.append([0.0, 0.0, 0.0, 0.0, 0.0])
        mt1 = []
        for x in mtt:
            for y in x:
                mt1.append(y)
        self.J5 = np.reshape(mt1, (6, 6), order="F")
        return self.J5

    def get_J6(self, q_1, q_2, q_3, q_4, q_5, q_6):
        #q_1, q_2, q_3, q_4, q_5, q_6 = self.degree2radians(q_1, q_2, q_3, q_4, q_5, q_6)
        t2 = cos(q_1)
        t3 = cos(q_5)
        t4 = sin(q_1)
        t5 = q_1 + q_2
        t6 = q_2 + q_3
        t7 = q_1 + q_5
        t15 = -q_1
        t16 = -q_2
        t17 = -q_5
        t8 = cos(t5)
        t9 = cos(t7)
        t10 = q_3 + t5
        t11 = q_4 + t6
        t12 = sin(t5)
        t13 = sin(t6)
        t14 = sin(t7)
        t24 = -t2
        t30 = q_1 + t16
        t31 = q_1 + t17
        t35 = t6 + t15
        t68 = t3* 6.123233995736766e-17
        t91 = t2* 3.749399456654644e-33
        t92 = t4* 3.749399456654644e-33
        t18 = cos(t10)
        t19 = cos(t11)
        t20 = q_4 + t10
        t21 = q_5 + t11
        t22 = sin(t10)
        t23 = sin(t11)
        t32 = cos(t30)
        t33 = cos(t31)
        t36 = sin(t30)
        t37 = sin(t31)
        t39 = t9/ 2.0
        t40 = t14/ 2.0
        t41 = cos(t35)
        t42 = t11 + t15
        t43 = t11 + t17
        t44 = sin(t35)
        t52 = t13* (9.0/ 5.0e+1)
        t57 = t9* (9.0/ 2.0e+2)
        t59 = t14* (9.0/ 2.0e+2)
        t62 = -t7 + t11
        t70 = t8* 1.5e-1
        t71 = t12* 1.5e-1
        t93 = -t91
        t25 = cos(t20)
        t26 = cos(t21)
        t27 = q_5 + t20
        t28 = sin(t20)
        t29 = sin(t21)
        t45 = -t39
        t46 = cos(t42)
        t47 = cos(t43)
        t48 = t17 + t20
        t49 = t15 + t21
        t50 = sin(t42)
        t51 = sin(t43)
        t58 = -t52
        t60 = t33/ 2.0
        t61 = t37/ 2.0
        t64 = cos(t62)
        t65 = sin(t62)
        t66 = t33* (9.0/ 2.0e+2)
        t67 = t37* (9.0/ 2.0e+2)
        t69 = t19* 6.123233995736766e-17
        t72 = -t70
        t73 = -t71
        t74 = t18* 9.000000000000001e-2
        t75 = t22* 9.000000000000001e-2
        t82 = t23* 8.000000000000001e-2
        t83 = t32* 1.5e-1
        t84 = t36* 1.5e-1
        t86 = t41* 8.999999999999999e-2
        t87 = t44* 8.999999999999999e-2
        t34 = sin(t27)
        t38 = cos(t27)
        t53 = cos(t48)
        t54 = cos(t49)
        t55 = sin(t48)
        t56 = sin(t49)
        t63 = -t60
        t76 = -t74
        t77 = -t75
        t78 = t26* 5.0e-1
        t79 = t29* 4.5e-2
        t81 = t47* 5.0e-1
        t85 = -t82
        t88 = -t86
        t89 = t51* 4.5e-2
        t94 = t25* 3.061616997868383e-17
        t95 = t28* 3.061616997868383e-17
        t97 = t50* 3.061616997868383e-17
        t98 = t46* 3.061616997868383e-17
        t101 = t25* 4.000000000000001e-2
        t102 = t28* 4.000000000000001e-2
        t122 = t64* 2.5e-1
        t123 = t65* 2.5e-1
        t124 = t46* 4.0e-2
        t125 = t50* 4.0e-2
        t129 = t64* 2.25e-2
        t130 = t65* 2.25e-2
        t80 = -t78
        t90 = -t89
        t96 = -t95
        t99 = -t97
        t100 = -t98
        t103 = t38* 2.25e-2
        t104 = t34* 2.25e-2
        t105 = t38* 2.5e-1
        t106 = t34* 2.5e-1
        t107 = -t101
        t108 = -t102
        t110 = t53* 2.5e-1
        t111 = t54* 2.5e-1
        t112 = t55* 2.5e-1
        t113 = t56* 2.5e-1
        t115 = t53* 2.25e-2
        t116 = t54* 2.25e-2
        t117 = t55* 2.25e-2
        t118 = t56* 2.25e-2
        t126 = -t122
        t127 = -t123
        t128 = -t124
        t131 = -t129
        t132 = -t130
        t109 = -t105
        t114 = -t112
        t119 = -t115
        t120 = -t117
        t121 = -t118
        t133 = t68 + t69 + t80 + t81 + 2.295845021658468e-49
        t134 = t40 + t61 + t92 + t96 + t99 + t106 + t113 + t114 + t127
        t135 = t45 + t63 + t93 + t94 + t100 + t109 + t110 + t111 + t126
        mtt = []
        mt1 = [t2* 9.35e-2 + t57 + t66 + t72 + t76 + t83 + t86 + t103 + t107 - t116 + t119 + t124 + t129]
        mtt.append(mt1)
        mtt.append([t4* 9.35e-2 + t59 + t67 + t73 + t77 + t84 - t87 + t104 + t108 + t118 + t120 - t125 + t132, 0.0, t4,
               t24])

        mtt.append([6.123233995736766e-17, t72 + t76 - t83 + t88 + t103 + t107 + t116 + t119 + t128 + t131,
               t73 + t77 - t84 + t87 + t104 + t108 + t120 + t121 + t125 + t130,
               t58 + t79 + t85 + t90 - sin(q_2)* (3.0/ 1.0e+1), t4, t24, 6.123233995736766e-17,
               t76 + t88 + t103 + t107 + t116 + t119 + t128 + t131, t77 + t87 + t104 + t108 + t120 + t121 + t125 + t130,
               t58 + t79 + t85 + t90, t4, t24])
        mtt.append([6.123233995736766e-17, t103 + t107 + t116 + t119 + t128 + t131, t104 + t108 + t120 + t121 + t125 + t130,
               t79 + t85 + t90])
        mtt.append([t4* 6.123233995736766e-17 - t28* 5.0e-1 - t50* 5.0e-1])
        mtt.append([t2* (-6.123233995736766e-17) + t25* 5.0e-1 - t46* 5.0e-1])
        mtt.append([t19 + 3.749399456654644e-33, t57 - t66 + t103 + t115 + t116 + t129,
               t59 - t67 + t104 + t117 + t121 + t132, t79 + t89 - sin(q_5)* 5.510910596163089e-18, t134, t135, t133,
               0.0, 0.0, 0.0, t134, t135, t133])
        mt1 = []
        for x in mtt:
            for y in x:
                mt1.append(y)
        self.J6 = np.reshape(mt1, (6, 6), order="F")
        return self.J6

    def get_Jcamera(self, q_1, q_2, q_3, q_4, q_5):
        #q_6 = 0
        #q_1, q_2, q_3, q_4, q_5, q_6 = self.degree2radians(q_1, q_2, q_3, q_4, q_5, q_6)
        #translated from matlab, will need to be changed because the kinematics measurement coule be different
        t2 = cos(q_1)
        t3 = cos(q_3)
        t4 = cos(q_5)
        t5 = sin(q_1)
        t6 = sin(q_3)
        t7 = sin(q_5)
        t8 = q_1 + q_5
        t9 = q_2 + q_3 + q_4
        t10 = -q_1
        t11 = -q_5
        t15 = np.pi / 2.0
        t12 = cos(t9)
        t13 = q_1 + t9
        t14 = -t2
        t17 = t8 + t9
        t19 = q_1 + t11
        t20 = -t15
        t21 = t3 * (9.0 / 5.0e+1)
        t22 = t4 / 2.0e+2
        t23 = t6 * (9.0 / 5.0e+1)
        t26 = t9 + t10
        t35 = -t8 + t9
        t16 = cos(t13)
        t18 = sin(t13)
        t24 = q_2 + t20
        t25 = q_4 + t20
        t29 = cos(t26)
        t32 = t11 + t13
        t33 = q_5 + t26
        t34 = sin(t26)
        t58 = t22 + 9.350000000000001e-2
        t27 = cos(t24)
        t28 = cos(t25)
        t30 = sin(t24)
        t31 = sin(t25)
        t36 = t27 * (3.0 / 1.0e+1)
        t37 = t30 * (3.0 / 1.0e+1)
        t38 = t22 * t28
        t39 = t22 * t31
        t40 = (t7 * t28) / 2.0e+2
        t41 = (t7 * t31) / 2.0e+2
        t42 = t4 * t28 * 3.061616997868383e-19
        t43 = t4 * t31 * 3.061616997868383e-19
        t44 = t7 * t28 * 3.061616997868383e-19
        t45 = t7 * t31 * 3.061616997868383e-19
        t48 = t28 * 1.354e-1
        t49 = t31 * 1.354e-1
        t46 = -t42
        t47 = -t45
        t50 = t39 + t44
        t51 = -t49
        t52 = t38 + t47
        t53 = t3 * t50
        t54 = t6 * t50
        t59 = t41 + t46 + t48
        t60 = t40 + t43 + t51
        t55 = t3 * t52
        t56 = t6 * t52
        t57 = -t54
        t61 = t3 * t59
        t62 = t3 * t60
        t63 = t6 * t59
        t64 = t6 * t60
        t65 = -t63
        t66 = t53 + t56
        t67 = t55 + t57
        t70 = -t27 * (t54 - t55)
        t71 = -t30 * (t54 - t55)
        t73 = t61 + t64
        t68 = t27 * t66
        t69 = t30 * t66
        t74 = t62 + t65
        t75 = t23 + t73
        t77 = t27 * t73
        t78 = t30 * t73
        t72 = -t69
        t76 = t21 + t74
        t79 = t27 * t74
        t80 = t30 * t74
        t81 = -t78
        t82 = t27 * t75
        t83 = t30 * t75
        t88 = t68 + t71
        t84 = t27 * t76
        t85 = t30 * t76
        t86 = -t83
        t89 = t70 + t72
        t90 = t77 + t80
        t91 = t79 + t81
        t87 = -t84
        t92 = t82 + t85
        t93 = t84 + t86
        t94 = t37 + t92
        t95 = t36 + t93
        et1 = t5 * 3.749399456654644e-33
        et2 = t18 * (-3.061616997868383e-17)
        et3 = t34 * (-3.061616997868383e-17) + sin(t8)   / 2.0
        et4 = sin(t17) * 2.5e-1 + sin(t19) / 2.0
        et5 = sin(t32) * (-2.5e-1)
        et6 = sin(t33) * 2.5e-1
        et7 = sin(t35) * (-2.5e-1)
        et8 = t2 * (-3.749399456654644e-33)
        et9 = t16 * 3.061616997868383e-17
        et10 = t29 * (-3.061616997868383e-17) - cos(t8) / 2.0
        et11 = cos(t17) * (-2.5e-1) - cos(t19) / 2.0
        et12 = cos(t32) * 2.5e-1
        et13 = cos(t33) * 2.5e-1
        et14 = cos(t35) * (-2.5e-1)
        et15 = t4 * 6.123233995736766e-17 + t12 * 6.123233995736766e-17 - cos(q_5 + t9) * 5.0e-1
        et16 = cos(t9 + t11) * 5.0e-1
        et17 = 2.295845021658468e-49
        mt1 = []
        mtt = []
        mt1.append([t2 * t58 + t2 * t94 * 6.123233995736766e-17 + t5 * t95,
               t5 * t58 + t5 * t94 * 6.123233995736766e-17 + t14 * t95, 0.0, t5, t14, 6.123233995736766e-17])
        mt1.append([t2 * t94 + t5 * t95 * 6.123233995736766e-17, t2 * t95 * (-6.123233995736766e-17) + t5 * t94,
               -t36 + t83 + t87, t5, t14, 6.123233995736766e-17])
        mt1.append([t5 * (t83 + t87) * (-6.123233995736766e-17) + t2 * t92,
               t2 * (t83 + t87) * 6.123233995736766e-17 + t5 * t92, t83 + t87, t5, t14, 6.123233995736766e-17])
        mt1.append([-t14 * t90 - t5 * (t78 - t79) * 6.123233995736766e-17,
               t5 * t90 + t2 * (t78 - t79) * 6.123233995736766e-17, t78 - t79])
        mt1.append([t5 * 6.123233995736766e-17 - t18 * 5.0e-1 - t34 * 5.0e-1])
        mt1.append([t2 * (-6.123233995736766e-17) + t16 * 5.0e-1 - t29 * 5.0e-1])
        mt1.append([t12 + 3.749399456654644e-33,
               -t14 * (t69 + t27 * (t54 - t55)) - (t5 * t7) / 2.0e+2 + t5 * t88 * 6.123233995736766e-17])
        mt1.append([t5 * (t69 + t27 * (t54 - t55)) + (t2 * t7) / 2.0e+2 - t2 * t88 * 6.123233995736766e-17,
               t7 * (-3.061616997868383e-19) - t68 + t30 * (t54 - t55), et1 + et2 + et3 + et4 + et5 + et6 + et7,
               et8 + et9 + et10 + et11 + et12 + et13 + et14, et15 + et16 + et17, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for x in mt1:
            for y in x:
                mtt.append(y)
        self.Jacobi_camera = np.reshape(mtt, (6, 6), order="F")
        return self.Jacobi_camera

    def get_T03(self, q_1, q_2, q_3):
        #q_4 = 0
        #q_5 = 0
        #q_6 = 0
        #q_1, q_2, q_3, q_4, q_5, q_6 = self.degree2radians(q_1, q_2, q_3, q_4, q_5, q_6)
        t2 = cos(q_1)
        t3 = cos(q_3)
        t4 = sin(q_1)
        t5 = sin(q_3)
        t6 = np.pi/ 2.0
        t7 = -t6
        t8 = q_2 + t7
        t9 = cos(t8)
        t10 = sin(t8)
        t11 = t2* t9
        t12 = t2* t10
        t13 = t4* t9
        t14 = t4* t10
        t15 = -t14
        t16 = t11* 6.123233995736766e-17
        t17 = t12* 6.123233995736766e-17
        t18 = t13* 6.123233995736766e-17
        t19 = t14* 6.123233995736766e-17
        t20 = -t19
        t21 = t12 + t18
        t22 = t13 + t17
        t24 = t15 + t16
        t23 = t11 + t20
        mt1 = [t3* t23 - t5* t21, t3* t22 + t5* t24, t3* t10 + t5* t9, 0.0, -t3* t21 - t5* t23,
               t3* t24 - t5* t22, t3* t9 - t5* t10, 0.0, t4, -t2, 6.123233995736766e-17, 0.0,
               t11* (-3.0/ 1.0e+1) + t14* 1.83697019872103e-17 - t3* t23* (9.0/ 5.0e+1) + t5* t21* (
                           9.0/ 5.0e+1)]
        mt2 = [t12* (-1.83697019872103e-17) - t13* (3.0/ 1.0e+1) - t3* t22* (9.0/ 5.0e+1) - t5* t24* (
                    9.0/ 5.0e+1),
               t10* (-3.0/ 1.0e+1) - t3* t10* (9.0/ 5.0e+1) - t5* t9* (9.0/ 5.0e+1) + 1.0/ 5.0, 1.0]

        mtt = []
        for x in mt1:
            mtt.append(x)
        for x in mt2:
            mtt.append(x)
        self.T03 = np.reshape(mtt, (4,4), order="F")

    def get_T3(self, q_3):
        #q_3 = q_3 * 180 / np.pi
        t2 = cos(q_3)
        t3 = sin(q_3)
        self.T3 = np.reshape(
            [t2, t3, 0.0, 0.0, -t3, t2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, t2* (-9.0/ 5.0e+1), t3* (-9.0/ 5.0e+1),
             0.0, 1.0], (4,4), order="F")

    def get_T4(self, q_4):
        t2 = np.pi / 2.0
        t3 = -t2
        t4 = q_4 + t3
        t5 = cos(t4)
        t6 = sin(t4)
        self.T4 = np.reshape([t5,t6,0.0,0.0,t6*(-6.123233995736766e-17),t5*6.123233995736766e-17,1.0,0.0,t6,-t5,6.123233995736766e-17,0.0,0.0,0.0,9.35e-2,1.0], (4,4), order="F")

    def get_T5(self, q_5):
        #q_5 = q_5 * 180 / np.pi
        t2 = cos(q_5)
        t3 = sin(q_5)
        self.T5 = np.reshape(
            [t2,t3,0.0,0.0,t3*(-6.123233995736766e-17),t2*6.123233995736766e-17,-1.0,0.0,-t3,t2,6.123233995736766e-17,0.0,0.0,0.0,2.0/2.5e+1,1.0], (4,4), order="F")

    def get_T6(self, q_6):
        #q_6 = q_6 * 180 / np.pi
        t2 = cos(q_6)
        t3 = sin(q_6)
        self.T6 = np.reshape([t2, t3, 0.0, 0.0, -t3, t2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 / 2.5e+1, 1.0], (4,4), order="F")

    def get_origin_pos(self, q):
        self.get_T03(q[0], q[1], q[2])
        self.get_T4(q[3])
        self.get_T5(q[4])
        self.get_T6(q[5])
        o0 = np.array([[0],[0],[0],[1]])
        o3 = self.T03 @ o0
        o4 = (self.T03 @ self.T4) @ o0
        o5 = (self.T03 @ self.T4 @ self.T5) @ o0
        o6 = (self.T03 @ self.T4 @ self.T5 @ self.T6) @ o0
        return [o3, o4, o5, o6]

    def check_collision(self, q, dq_vs):
        #self.padding
        '''
         1. check if the origins are close to z = 0
            a. if so decrease the velocity if and only if the velocity of the origins are still approahing z = 0
         2.check if the joints are close to joint limit
            a. if so decrease the velocities
        '''
        status = [[],[]]
        origin_pos = self.get_origin_pos(q)
        for i in range(0,4):
            ground_distance = origin_pos[i][2] - self.collision_radius[i]
            #check if the robot arm is about to collide with z = 0

            if ground_distance <= self.padding[0]:
                status[0] = 1
                status[1].append(i+3)
                jacobian = self.list_of_functions[i](q[0], q[1], q[2], q[3], q[4], q[5])
                origin_vel = jacobian @ dq_vs

                if (origin_vel[2] < 0):
                    #decrese the velocity: v = 40*Vcurrent*(z - self.padding[1])
                    newVz = 40*origin_vel[2]*(ground_distance-self.padding[1])
                    origin_vel[2] = newVz
                    try:
                        invJ = np.linalg.inv(jacobian)
                    except:
                        invJ = np.linalg.pinv(jacobian)
                    #get the joint velocities of the new velocity
                    joint_vel = invJ @ origin_vel
                    for j in range(i+1,6):
                        dq_vs[j] = joint_vel[j]
        #check if the robot arm is about to collide with itself, if so set it to 0
        for k in range(len(self.self_collision_constraint)):
            index = np.flatnonzero(self.self_collision_constraint[k] != 0)
            #print("sum is: ", abs(sum(q[index])*180/np.pi), "self collision is: ", abs(sum(self.self_collision_constraint[k])) - self.angle_padding)
            print("sign difference is: ", np.sign(dq_vs[index]))
            if abs(sum(q[index])*180/np.pi) > (abs(sum(self.self_collision_constraint[k])) - self.angle_padding) and all(np.sign(q[index[0]]) == np.sign(dq_vs[index])):
                if status == 1:
                    status[0] = 3
                    status[1].append(k+1)
                else:
                    status[0] = 2
                    status[1].append(k+1)
                #if the combination of angles are large enough (which means incoming collision) and none of them are moving away
                dq_vs[index] = 0

        return status, origin_pos, dq_vs

    def vector2rpy(self, vector):
        """
        :param vector: the vector z axis desired to be pointed to, should be an np array
        :return: rpy angles
        """
        vector = vector / np.linalg.norm(vector)
        rx = -asin(vector[1])
        ry = asin(vector[0] / cos(rx))
        check = cos(rx) * cos(ry)
        if np.sign(check) != np.sign(vector[2]):
            ry = np.pi - ry
        return np.array([[rx],[ry],[0]])
    def determine_orientation(self, ball_pos, ball_vel, ball_detected, predicted_pos):
        """
        :param ball_pos: abs position of the ball, should keep the LAST VALUE of the ball if not in frame, should be an np array
        :param ball_vel: abs velocity of the ball, should keep the LAST VALUE of the ball if not in frame, should be an np array
        :param ball_detected: whether the ball is in the frame or not
        :param predicted_pos: predicted position from the kalman filter, if there isn't oen then it should be [0,0,0]
        :return: orientation rpy angles
        three conditions:
        1. when the ball is too high up and the z velocity >=0 : return orientation which vector is opposite of the ball's vector
        2. when the ball is too high up, z velocity <0 and it is in front of the arm: same as 1
        3. when the ball is too high up, z velocity <0 and it is behind or at the arm: same as 1
        4. when the ball is in the z range and in front of the arm: orientation will be [0,0,0]
        5. when the ball is in the z range and behind the arm: orientation will be [0,0,0]
        6 when the ball is too low (z<=0 or a small number): stop the arm, which is already done in visual servoing
        7. if the ball went out of the page yet z velocity <0: still go down unless it is below padding
        8. if the ball went out of the page yet z velocity >0: same as 6
        """
        
        ball_pos = ball_pos[0:3]
        if (ball_detected == True and ball_pos[2] + self.max_height_padding >= self.max_horizontal_look_height) or (ball_detected == False and ball_vel[2] > 0 and predicted_pos[0] == -100):
            return self.vector2rpy(ball_pos * (-1))
        elif (ball_detected == True and ball_pos[2] - self.max_height_padding <= self.min_height_padding) or (ball_detected == False and ball_vel[2] < 0 and predicted_pos[0] == -100):
            return self.vector2rpy((ball_pos - np.array([[0],[0],[self.min_height_padding]])) * (-1))
        elif ball_detected == False and ball_vel[2] > 0 and predicted_pos[0] != -100:
            return self.vector2rpy(predicted_pos * (-1))
        elif ball_detected == False and ball_vel[2] < 0 and predicted_pos[0] != -100:
            return self.vector2rpy((predicted_pos - np.array([[0],[0],[self.min_height_padding]])) * (-1))
        else:
            return np.array([[0],[0],[0]])

    def visual_servoing(self, q, dq, goal_coord, current_coord, ball_vel, ball_detected, predicted_pos):
        '''
        :param q: 6 x 1, angle state of the robot joints
        :param goal_coord: 6 x 1 (3 for coordinates, three for orientation), coordinates where the ball's centroid should be in the camera frame (eg. at the centre)
        :param current_coord: 6 x 1, coordinates where the ball's centroid is at right now
        :param ball_vel: abs velocity of the ball, should keep the LAST VALUE of the ball if not in frame, should be an np array
        :param ball_detected: whether the ball is in the frame or not
        :param predicted_pos: predicted position from the kalman filter, if there isn't oen then it should be [0,0,0]
        :return: joint_vel: 6 x 1, the velocity for the robot's joints to be at for visual servoing to work
        '''
        self.desired_orientation = self.determine_orientation(current_coord, ball_vel, ball_detected, predicted_pos)
        goal_coord[3:] = self.desired_orientation
        error = goal_coord - current_coord

        # try to inverse the jacobian, but if it is singular for some reason, use pseudo inverse
        self.get_Jcamera(q[0], q[1], q[2], q[3], q[4])
        try:
            invJ = np.linalg.inv(self.Jacobi_camera)
        except:
            invJ = np.linalg.pinv(self.Jacobi_camera)
        """
        TODO
        1. need to cap out velocity to avoid singularity
        2. need to avoid collision and joint limits in visual servoing
        3. Need to figure out an algorithm to determine the orientation
        """
        joint_vel = self.speed_reduce * invJ @ (self.clamp * error)
        print("joint vel at first is: ", joint_vel)
        status, origin_pos, joint_vel = self.check_collision(q, joint_vel)
        return status, origin_pos, joint_vel


