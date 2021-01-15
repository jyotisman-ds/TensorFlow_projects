let model_beans;
var tensorFeature
var loadFile = function(event) {
	image = document.getElementById('output');
    image.src = URL.createObjectURL(event.target.files[0]);
    image.onload = () => {
    tensorFeature = tf.browser.fromPixels(image).resizeBilinear([150,150]).expandDims();
    }
};

let isPredicting = false;
    
async function loadBeans() {
    const MODEL_URL = 'http://127.0.0.1:8887/model.json';
    const model_beans = await tf.loadLayersModel(MODEL_URL);
    return tf.model({inputs: model_beans.inputs, outputs: model_beans.output});
}
    
async function predict(){

    if (isPredicting) {
            const predictedClass = tf.tidy(() => {
            const predictions = model_beans.predict(tensorFeature);
            return predictions.as1D().argMax();
            });
            const classId = (await predictedClass.data())[0];
        
            switch(classId){
		    case 0:
			 predictionText = "Angular leaf spot(0)";
            break;
		    case 1:
			 predictionText = "Bean rust(1)";
			break;
		    case 2:
			 predictionText = "Healthy(2)";
			break;
            }
            document.getElementById("prediction").innerText = predictionText;
    
            predictedClass.dispose();
    }
}
    
function startPredicting(){
	isPredicting = true;
	predict();
}


async function init(){
	   model_beans = await loadBeans();
       console.log(model_beans.summary());
}

init();