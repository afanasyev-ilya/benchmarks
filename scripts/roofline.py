#!/usr/bin/python


import plotly
import plotly.graph_objs as go
import bisect
import sys

tmp_data_prefix = "prof_data/"

last_band = "L1"
last_perf = "float_vector_FMA"

kunpeng_characteristics = {"bandwidths": {"DRAM": 187, "L3": 1060, "L2": 1800, "L1": 2200},
                                         "peak_performances": {"float_no_vector_noFMA": 124,
                                                               "float_vector_noFMA": 499,
                                                               "float_vector_FMA": 1996}}  # GFLOP/s

intel_xeon_characteristics = {"bandwidths": {"DRAM": 127, "L3": 1, "L2": 1, "L1": 1},
                              "peak_performances": {"float_no_vector_noFMA": 1,
                                                 "float_vector_noFMA": 1,
                                                 "float_vector_FMA": 1}}  # GFLOP/s

amd_epyc_characteristics = {"bandwidths": {"DRAM": 204, "L3": 1, "L2": 1, "L1": 1},
                            "peak_performances": {"float_no_vector_noFMA": 1,
                                                    "float_vector_noFMA": 1,
                                                    "float_vector_FMA": 1}}  # GFLOP/s

x_data_first = 1.0 / 256.0
x_data_last = 1024


class RooflinePlotter:
    def __init__(self, name, platform_characteristics):
        self.platform_characteristics = platform_characteristics
        self.total_execution_time = 0.0
        self.name = name

    def get_compute_roof(self, x, roof_bandwidth, perf_key):
        return min(self.platform_characteristics["peak_performances"][perf_key], roof_bandwidth * x)

    def get_bandwidth_roof(self, x, roof_bandwidth):
        return roof_bandwidth * x

    def create_x_data(self, x_min, x_max, additional_x_points):
        x_array = []
        current = x_min
        prev = 0
        while current <= x_max:
            x_array.append(current)
            prev = current
            current += 0.1 # TODO 0.01

        return x_array

    def generate_CARM_roof_plots(self, additional_x_points):
        x_data = self.create_x_data(x_data_first, x_data_last, additional_x_points)
        data = []

        max_band = self.platform_characteristics["bandwidths"][last_band]
        max_perf = self.platform_characteristics["peak_performances"][last_perf]
        x_intersection = max_perf/max_band

        for perf_key in self.platform_characteristics["peak_performances"]:
            y_data = []
            cur_perf = self.platform_characteristics["peak_performances"][perf_key]
            for x in x_data:
                y_data.append(cur_perf)
            data.append(go.Scatter(x=x_data, y=y_data, mode="lines", name=str(perf_key),
                                   text=[perf_key], showlegend=False))
            x_data_name=[x_intersection]
            y_data_name=[cur_perf]
            data.append(go.Scatter(x=x_data_name, y=y_data_name, mode="lines+text", name=str(perf_key),
                                   text=[perf_key], textposition="top right", showlegend=False))

        for mem_key in self.platform_characteristics["bandwidths"]:
            y_data = []
            cur_band = self.platform_characteristics["bandwidths"][mem_key]
            x_intersection = max_perf / self.platform_characteristics["bandwidths"][mem_key]
            for x in x_data:
                if x < x_intersection:
                    y_data.append(self.get_bandwidth_roof(x, cur_band))
            data.append(go.Scatter(x=x_data, y=y_data, mode="lines", name=str(mem_key), text=[mem_key],
                                   showlegend=False))
            x_data_name = [(x_intersection - x_data_first)/64 + x_data_first]
            y_data_name = [self.get_bandwidth_roof(x_data_name[0], cur_band)]
            data.append(go.Scatter(x=x_data_name, y=y_data_name, mode="lines+text", name=str(mem_key), text=[mem_key],
                                   textposition="top center", showlegend=False))

        return data

    def get_point_description_text(self, profiling_data):
        point_description_text = profiling_data["name"] + "</br>"
        x = float(profiling_data["ops_per_byte"])

        # memory bound case
        top_roof_val = 0
        top_roof_name = ""
        closest_roof_val = 0
        closest_roof_name = ""
        point_val = profiling_data["gflops"]
        mem_key = "DRAM"
        perf_key = "float_no_vector_noFMA"

        min_distance = self.platform_characteristics["peak_performances"][perf_key]
        roof_val = self.get_compute_roof(self.platform_characteristics["bandwidths"][mem_key], x, "float_no_vector_noFMA") # TODO
        if roof_val >= top_roof_val:
            top_roof_val = roof_val
            top_roof_name = mem_key
        if (roof_val >= point_val) and (min_distance >= abs(roof_val - point_val)):
            min_distance = abs(roof_val - point_val)
            closest_roof_val = roof_val
            closest_roof_name = perf_key

        # compute bound case
        closest_roof_name = "Peak performance"

        if closest_roof_val > 0:
            distance_from_closest_roof = 100.0 * float(point_val) / float(closest_roof_val)
        else:
            distance_from_closest_roof = 100
        if top_roof_val > 0:
            distance_from_top_roof = 100.0 * float(point_val) / float(top_roof_val)
        else:
            distance_from_top_roof = 100

        if top_roof_val != closest_roof_val:  # print closest roof only if it is different from top toof
            point_description_text += closest_roof_name + " - closest roof: " + str(distance_from_closest_roof) + \
                                      "% </br>"
        point_description_text += top_roof_name + " -  top roof: " + str(distance_from_top_roof) + "% </br>"
        return point_description_text

    def generate_roofline_point_plot(self, profiling_data):
        point_description_text = self.get_point_description_text(profiling_data)
        point_trace = go.Scatter(
            x=[profiling_data["ops_per_byte"], profiling_data["ops_per_byte"]],
            y=[0, profiling_data["gflops"]],
            name=profiling_data["name"],
            mode='markers',
            text=['', profiling_data["name"], ''],
            textposition='top center'
        )
        return point_trace

    def get_profiling_points_x_data(self, profiling_data_array):
        additional_x_points = []
        for profiling_data in profiling_data_array:
            additional_x_points.append(profiling_data["ops_per_byte"])
        return additional_x_points

    def draw_plot(self, profiling_data_array, sockets):
        profiling_points_x_data = [] #self.get_profiling_points_x_data(profiling_data_array)
        plots_data = self.generate_CARM_roof_plots(profiling_points_x_data)

        for profiling_data in profiling_data_array:
            plots_data.append(self.generate_roofline_point_plot(profiling_data))

        y_title = "GFLOP/s"
        xaxis = dict(autorange=True, showgrid=True, zeroline=True, showline=True, ticks='',
                     showticklabels=True, type='log', title='Arithmetic Intensity')

        yaxis = dict(autorange=True, showgrid=True, zeroline=True, showline=True, ticks='',
                     showticklabels=True, type='log', title=y_title)

        current_file_name='./../benchmarks_new/output/roofline_'+sockets+'.html'
        #current_file_name = 'roofline_' + sockets + '.html'
        plotly.offline.plot({
            "data": plots_data,
            "layout": go.Layout(title=self.name, xaxis=xaxis, yaxis=yaxis)
        }, filename=current_file_name)


def generate_roofline_from_profiling_data(file_name, roofline_name):
    # read CMD file
    profiling_file = open(file_name, 'r')
    profiling_data = []

    line_pos = 0
    for line in profiling_file:
        line_pos += 1
        if line_pos == 1:
            platform_characteristics = kunpeng_characteristics
            continue
        if line.startswith("#") or line.startswith("//"):
            continue
        line_split = line.split(",")
        name = line_split[0]
        ops_number = float(line_split[1])
        ops_per_byte = float(line_split[2])
        profiling_data.append({"name": name,
                               "ops_per_byte": ops_per_byte,
                               "gflops": ops_number})
    profiling_file.close()

    # initialize and draw roofline
    roofline = RooflinePlotter(roofline_name, platform_characteristics)
    roofline.draw_plot(profiling_data, "single_socket")


