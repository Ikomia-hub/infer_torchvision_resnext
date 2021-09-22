from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from infer_torchvision_resnext.infer_torchvision_resnext_process import ResnextFactory
        # Instantiate process object
        return ResnextFactory()

    def getWidgetFactory(self):
        from infer_torchvision_resnext.infer_torchvision_resnext_widget import ResnextWidgetFactory
        # Instantiate associated widget object
        return ResnextWidgetFactory()
