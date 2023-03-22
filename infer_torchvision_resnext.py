from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        from infer_torchvision_resnext.infer_torchvision_resnext_process import ResnextFactory
        # Instantiate process object
        return ResnextFactory()

    def get_widget_factory(self):
        from infer_torchvision_resnext.infer_torchvision_resnext_widget import ResnextWidgetFactory
        # Instantiate associated widget object
        return ResnextWidgetFactory()
