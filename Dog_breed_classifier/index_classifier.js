let model_dog_breed;
var tensorFeature
var loadFile = function(event) {
	image = document.getElementById('output');
  image.src = URL.createObjectURL(event.target.files[0]);
  image.onload = () => {
  tensorFeature = tf.browser.fromPixels(image).resizeBilinear([240,240]).expandDims();
	tensorFeature = tensorFeature.div(127.5);
	tensorFeature = tensorFeature.add(-1);
  }
};

let isPredicting = false;

async function loadDB() {
    const MODEL_URL = 'http://127.0.0.1:8887/model.json';
    const model_dog_breed = await tf.loadLayersModel(MODEL_URL);
    return tf.model({inputs: model_dog_breed.inputs, outputs: model_dog_breed.output});
}

async function predict(){

    if (isPredicting) {
            const predictedClass = tf.tidy(() => {
            const predictions = model_dog_breed.predict(tensorFeature);
            return predictions.as1D();
            });
            const classId = (await predictedClass.argMax().data())[0];
            const labels_list = ['chihuahua',
 						 'japanese_spaniel',
 					 	 'maltese_dog',
 					 	 'pekinese',
 					 	 'tzu',
 					     'blenheim_spaniel',
 					     'papillon',
 					     'toy_terrier',
 					     'rhodesian_ridgeback',
 					     'afghan_hound',
 					 	 'basset',
 					 	 'beagle',
 					 	 'bloodhound',
 					 	 'bluetick',
 					 	 'tan_coonhound',
						 'walker_hound',
 					     'english_foxhound',
 					     'redbone',
 					 	 'borzoi',
 					 	 'irish_wolfhound',
 					 	 'italian_greyhound',
 					 	 'whippet',
 					 	 'ibizan_hound',
 					 	 'norwegian_elkhound',
 					 	 'otterhound',
 					 	 'saluki',
 					     'scottish_deerhound',
 					 	 'weimaraner',
 					 	 'staffordshire_bullterrier',
 					 	 'american_staffordshire_terrier',
 					     'bedlington_terrier',
 					     'border_terrier',
 					 	 'kerry_blue_terrier',
						 'irish_terrier',
						 'norfolk_terrier',
						 'norwich_terrier',
						 'yorkshire_terrier',
						 'haired_fox_terrier',
						 'lakeland_terrier',
						 'sealyham_terrier',
						 'airedale',
						 'cairn',
						 'australian_terrier',
						 'dandie_dinmont',
						 'boston_bull',
						 'miniature_schnauzer',
						 'giant_schnauzer',
						 'standard_schnauzer',
						 'scotch_terrier',
						 'tibetan_terrier',
						 'silky_terrier',
						 'coated_wheaten_terrier',
						 'west_highland_white_terrier',
						 'lhasa',
						 'coated_retriever',
						 'coated_retriever',
						 'golden_retriever',
						 'labrador_retriever',
						 'chesapeake_bay_retriever',
						 'haired_pointer',
						 'vizsla',
						 'english_setter',
						 'irish_setter',
						 'gordon_setter',
						 'brittany_spaniel',
						 'clumber',
						 'english_springer',
						 'welsh_springer_spaniel',
						 'cocker_spaniel',
						 'sussex_spaniel',
						 'irish_water_spaniel',
						 'kuvasz',
						 'schipperke',
						 'groenendael',
						 'malinois',
						 'briard',
						 'kelpie',
						 'komondor',
						 'old_english_sheepdog',
						 'shetland_sheepdog',
						 'collie',
						 'border_collie',
						 'bouvier_des_flandres',
						 'rottweiler',
						 'german_shepherd',
						 'doberman',
						 'miniature_pinscher',
						 'greater_swiss_mountain_dog',
						 'bernese_mountain_dog',
						 'appenzeller',
						 'entlebucher',
						 'boxer',
						 'bull_mastiff',
						 'tibetan_mastiff',
						 'french_bulldog',
						 'great_dane',
						 'saint_bernard',
						 'eskimo_dog',
						 'malamute',
						 'siberian_husky',
						 'affenpinscher',
						 'basenji',
						 'pug',
						 'leonberg',
						 'newfoundland',
						 'great_pyrenees',
						 'samoyed',
						 'pomeranian',
						 'chow',
						 'keeshond',
						 'brabancon_griffon',
						 'pembroke',
						 'cardigan',
						 'toy_poodle',
						 'miniature_poodle',
						 'standard_poodle',
						 'mexican_hairless',
						 'dingo',
						 'dhole',
						 'african_hunting_dog'];

            document.getElementById("prediction").innerText = "The dog's breed is : " + labels_list[classId] + " with a probability of : " + predictedClass.arraySync()[classId];

            predictedClass.dispose();
    }
}

function startPredicting(){
	isPredicting = true;
	predict();
}


async function init(){
	   model_dog_breed = await loadDB();
     console.log(model_dog_breed.summary());
}

init();
