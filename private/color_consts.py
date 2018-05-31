# Copyright (c) 2018 Nguyen Viet Giang. All rights reserved.
import os.path
import string
import sys

def color(output_path):
    y0 = (6.0 / 29.0)*(6.0 / 29.0)*(6.0 / 29.0)
    a = (29.0 / 3.0)*(29.0 / 3.0)*(29.0 / 3.0)
    un = 0.197833
    vn = 0.468331
    nrm = 1.0 / 255.0
    mr0 = 0.430574*nrm; mr1 = 0.222015*nrm; mr2 = 0.020183*nrm
    mg0 = 0.341550*nrm; mg1 = 0.706655*nrm; mg2 = 0.129553*nrm
    mb0 = 0.178325*nrm; mb1 = 0.071330*nrm; mb2 = 0.939180*nrm
    maxi = 1.0 / 270; minu = -88*maxi; minv = -134*maxi
    ltable = list()
    for i in range(1025):
        y = i / 1024.0
        l = 116.0*pow(y, 1.0/3.0) - 16 if y > y0 else y*a
        ltable.append(l*maxi)
    for i in range(1025, 1064):
        ltable.append(ltable[-1])
    output = "typedef struct color_consts_s {\n"
    output += "    float y0, a, un, vn;\n"
    output += "    cl_float4 mr, mg, mb;\n"
    output += "    float maxi, minu, minv;\n"
    output += "} color_consts_t;\n\n"
    output += "static const float color_ltable[] = {\n"
    for i in ltable:
        output += "    %ff,\n" % i
    output += "};\n\n"
    output += "static const color_consts_t color_consts = {\n"
    output += "    %ff, %ff, %ff, %ff,\n" % (y0, a, un, vn)
    output += "    { %ff, %ff, %ff, 0 },\n" % (mr0, mr1, mr2)
    output += "    { %ff, %ff, %ff, 0 },\n" % (mg0, mg1, mg2)
    output += "    { %ff, %ff, %ff, 0 },\n" % (mb0, mb1, mb2)
    output += "    %ff, %ff, %ff\n" % (maxi, minu, minv)
    output += "};\n"
    with open(output_path, "w") as output_file:
        output_file.write(output)

if __name__ == '__main__':
    color(sys.argv[1])