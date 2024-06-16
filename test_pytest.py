import creating_model as cm

def test_RemovedPrivateDataFromInput():
    forbiddenData = ['Customer Name', 'Customer e-mail', 'Country']
    input, output = cm.RemoveIrrelevantData(cm.GetData())
    assert not any([item in input.columns for item in forbiddenData])
    
def test_RemovedPrivateDataFromOutput():
    forbiddenData = ['Customer Name', 'Customer e-mail', 'Country']
    input, output = cm.RemoveIrrelevantData(cm.GetData())
    assert not any([item in output.columns for item in forbiddenData])