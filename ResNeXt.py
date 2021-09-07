from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class ResNeXt(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from ResNeXt.ResNeXt_process import ResNeXtProcessFactory
        # Instantiate process object
        return ResNeXtProcessFactory()

    def getWidgetFactory(self):
        from ResNeXt.ResNeXt_widget import ResNeXtWidgetFactory
        # Instantiate associated widget object
        return ResNeXtWidgetFactory()
