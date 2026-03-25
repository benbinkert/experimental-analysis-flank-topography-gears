from surfalize import Surface

from Code.Plots import peaks_and_distances_profile, peaks_more
from Code.Unterprogramme import plot_surface_with_oblique_line
from OrthogonaleLinie import *
from Hilfslinien import *
from scipy.io import loadmat


def tool():
    """
    filepath = '/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/ErodierteProben/WSP00_L1_S1.nms'
    surface1 = Surface.load(filepath)
    surface1.level()
    surface1 = surface1.detrend_polynomial(degree=2)
    #profile_erodiert = surface1.get_vertical_profile(x=5000)
    #surface1.show(False)
    #profile_erodiert.show()
    surface = Surface.load('/Users/benbinkert/PycharmProjects/Bachelorarbeit//Data/Simulation/WST_TOPO0_L1_Rechts.sdf')
    surface_cut = surface.crop((0, surface.width_um, 150, 3200))
    surface_cut = surface_cut.level()
    surface_cut.show()


    #result = draw_multiple_lines_and_measure(surface_cut)
    #result = draw_parallelized_lines_and_user_normal(surface1, show_profile=True)
    result = draw_parallelized_lines_and_user_normal_sim(surface, show_profile=True)


    surface2 = Surface.load("/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben/WSP03_L1_S1.nms")
    surface2.show()
    surface2 = surface2.level()
    surface2 = Surface.detrend_polynomial(surface2,2)
    surface2 = surface2.threshold(threshold=(0.25, 0.25))
    result = measure_horizontal_distance(surface2)
    #result = draw_multiple_lines_and_measure(surface2)
    #result = draw_parallelized_lines_and_user_normal(surface2, show_profile=True) 4444, 0,4976, 789
    #result = draw_parallelized_lines_and_user_normal_sim(surface, show_profile=True)


    filepath = '/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L1_S1.nms'
    surface_Erodiert = Surface.load(filepath)
    surface_Erodiert = surface_Erodiert.level()
    surface_Erodiert = Surface.detrend_polynomial(surface_Erodiert,2)
    surface_Erodiert = surface_Erodiert.threshold(threshold=(0.25, 0.25))
    surface_Erodiert = surface_Erodiert.fill_nonmeasured_rowwise_linear()
    surface_Erodiert = surface_Erodiert.filter(filter_type='lowpass', cutoff=1.6)
    surface_Erodiert = surface_Erodiert.fill_nonmeasured_rowwise_linear()
    surface = surface_Erodiert.crop((0,100,310,410))
    #result = draw_parallelized_lines_and_user_normal(surface)   #58, 100,70, 0
    """

    surface2 = Surface.load("/Data/Nanofocus/WSP03/ErodierteProben/WSP03_L1_S1.nms")
    #surface2.show()
    surface2 = surface2.level()
    surface2 = Surface.detrend_polynomial(surface2,2)
    surface2 = surface2.threshold(threshold=(0.25, 0.25))
    surface2 = surface2.fill_nonmeasured_rowwise_linear()
    surface2 =surface2.fill_nonmeasured_rowwise_linear()
    surface = surface2.crop((4250, 6000,470, 750))
    surface.fill_nonmeasured_rowwise_linear()
    surface = surface.crop((0, surface.width_um, 0,210))
    surface = surface.filter(filter_type="lowpass", cutoff=20)
    surface.show()

    #result = plot_surface_with_oblique_line_filter(surface,812, 0,958, 209,step_label_um=100,show_profile=  True)
   #surface.show()
    result = draw_multiple_lines_and_measure(surface)
    #result = draw_parallelized_lines_and_user_normal(surface)




if __name__ == "__main__":
   tool()
