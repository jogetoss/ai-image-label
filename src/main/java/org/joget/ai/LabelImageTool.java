package org.joget.ai;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.joget.apps.app.model.AppDefinition;
import org.joget.apps.app.service.AppService;
import org.joget.apps.app.service.AppUtil;
import org.joget.apps.form.model.Element;
import org.joget.apps.form.model.Form;
import org.joget.apps.form.model.FormData;
import org.joget.apps.form.model.FormRow;
import org.joget.apps.form.model.FormRowSet;
import org.joget.apps.form.service.FileUtil;
import org.joget.apps.form.service.FormUtil;
import org.joget.commons.util.LogUtil;
import org.joget.plugin.base.DefaultApplicationPlugin;
import org.joget.plugin.base.PluginManager;
import org.joget.workflow.model.WorkflowAssignment;
import org.joget.workflow.model.service.WorkflowManager;
import org.springframework.beans.BeansException;
import org.tensorflow.Tensor;
import org.tensorflow.examples.LabelImage;

public class LabelImageTool extends DefaultApplicationPlugin {

    @Override
    public String getName() {
        return "AI Label Image Tool";
    }

    @Override
    public String getVersion() {
        return "6.0.0";
    }

    @Override
    public String getDescription() {
        return "AI Label Image Tool";
    }

    @Override
    public String getLabel() {
        return "AI Label Image Tool";
    }

    @Override
    public String getClassName() {
        return getClass().getName();
    }

    @Override
    public String getPropertyOptions() {
        AppDefinition appDef = AppUtil.getCurrentAppDefinition();
        String appId = appDef.getId();
        String appVersion = appDef.getVersion().toString();
        Object[] arguments = new Object[]{appId, appVersion};
        String json = AppUtil.readPluginResource(getClass().getName(), "/properties/labelImageTool.json", arguments, true, null);
        return json;
    }

    @Override
    public Object execute(Map properties) {
        String label = "NA";
        Float probability = 0.0f;
        
        try {
            // load trained TensorFlow model
            AppService appService = (AppService)AppUtil.getApplicationContext().getBean("appService");
            PluginManager pluginManager = (PluginManager)AppUtil.getApplicationContext().getBean("pluginManager");
            byte[] graphDef = IOUtils.toByteArray(pluginManager.getPluginResource(getClassName(), "/properties/tensorflow_inception_graph.pb"));
            List<String> labels = IOUtils.readLines(pluginManager.getPluginResource(getClassName(), "/properties/imagenet_comp_graph_label_strings.txt"), "UTF-8");

            // get record ID
            String recordId;
            WorkflowAssignment wfAssignment = (WorkflowAssignment) properties.get("workflowAssignment");
            if (wfAssignment != null) {
                recordId = appService.getOriginProcessId(wfAssignment.getProcessId());
            } else {
                recordId = (String)properties.get("recordId");
            }
            
            // get file upload image
            String formDefId = (String)properties.get("formDefId");
            String fileUploadId = (String) properties.get("fileUploadId");
            AppDefinition appDef = (AppDefinition)properties.get("appDef");
            FormData formData = new FormData();
            formData.setPrimaryKeyValue(recordId);
            Form loadForm = appService.viewDataForm(appDef.getId(), appDef.getVersion().toString(), formDefId, null, null, null, formData, null, null);
            Element el = FormUtil.findElement(fileUploadId, loadForm, formData);
            File file = FileUtil.getFile(FormUtil.getElementPropertyValue(el, formData), loadForm, recordId);
            byte[] imageBytes = FileUtils.readFileToByteArray(file);

            // match image against trained TensorFlow model
            Tensor image = LabelImage.constructAndExecuteGraphToNormalizeImage(imageBytes);
            float[] labelProbabilities = LabelImage.executeInceptionGraph(graphDef, image);
            int bestLabelIdx = LabelImage.maxIndex(labelProbabilities);
            label = labels.get(bestLabelIdx);
            probability = labelProbabilities[bestLabelIdx] * 100f;
            
            // store results into form
            String formLabelMapping = (String) properties.get("formLabelMapping");
            String formProbabilityMapping = (String) properties.get("formProbabilityMapping");
            FormRowSet rowSet = appService.loadFormData(loadForm, recordId);
            if (!rowSet.isEmpty()) {
                FormRow row = rowSet.get(0);
                row.setProperty(formLabelMapping, label);
                row.setProperty(formProbabilityMapping, probability + "");
                appService.storeFormData(appDef.getId(), appDef.getVersion().toString(), formDefId, rowSet, null);
            }
            
            // store results into workflow variables
            String wfVariableLabelMapping = (String)properties.get("wfVariableLabelMapping");
            if (wfAssignment != null && wfVariableLabelMapping != null && !wfVariableLabelMapping.trim().isEmpty()) {
                WorkflowManager workflowManager = (WorkflowManager)AppUtil.getApplicationContext().getBean("workflowManager");
                workflowManager.activityVariable(wfAssignment.getActivityId(), wfVariableLabelMapping, label);
            }
            String wfVariableProbabilityMapping = (String)properties.get("wfVariableProbabilityMapping");
            if (wfAssignment != null && wfVariableProbabilityMapping != null && !wfVariableProbabilityMapping.trim().isEmpty()) {
                WorkflowManager workflowManager = (WorkflowManager)AppUtil.getApplicationContext().getBean("workflowManager");
                workflowManager.activityVariable(wfAssignment.getActivityId(), wfVariableProbabilityMapping, probability + "");
            }
            
        } catch(IOException | BeansException e) {
            LogUtil.error(getClassName(), e, e.getMessage());
        }

        LogUtil.info(getClassName(), String.format("BEST MATCH: %s (%.2f%% likely)", label, probability));
        
        return null;
    }
    
}
