let model_dog_breed;
var tensorFeature
var loadFile = function(event) {
	image = document.getElementById('output');
  image.src = URL.createObjectURL(event.target.files[0]);
  image.onload = () => {
  tensorFeature = tf.browser.fromPixels(image).resizeBilinear([150,150]).expandDims();
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
            return predictions.as1D().argMax();
            });
            const classId = (await predictedClass.data())[0];
						const labels_list = [b'chihuahua',
 						 b'japanese_spaniel',
 					 	 b'maltese_dog',
 					 	 b'pekinese',
 					 	 b'tzu',
 					   b'blenheim_spaniel',
 					   b'papillon',
 					   b'toy_terrier',
 					   b'rhodesian_ridgeback',
 					   b'afghan_hound',
 					 	 b'basset',
 					 	 b'beagle',
 					 	 b'bloodhound',
 					 	 b'bluetick',
 					 	 b'tan_coonhound',
						 b'walker_hound',
 					   b'english_foxhound',
 					   b'redbone',
 					 	 b'borzoi',
 					 	 b'irish_wolfhound',
 					 	 b'italian_greyhound',
 					 	 b'whippet',
 					 	 b'ibizan_hound',
 					 	 b'norwegian_elkhound',
 					 	 b'otterhound',
 					 	 b'saluki',
 					   b'scottish_deerhound',
 					 	 b'weimaraner',
 					 	 b'staffordshire_bullterrier',
 					 	 b'american_staffordshire_terrier',
 					   b'bedlington_terrier',
 					   b'border_terrier',
 					 	 b'kerry_blue_terrier',
						 b'irish_terrier',
						 b'norfolk_terrier',
						 b'norwich_terrier',
						 b'yorkshire_terrier',
						 b'haired_fox_terrier',
						 b'lakeland_terrier',
						 b'sealyham_terrier',
						 b'airedale',
						 b'cairn',
						 b'australian_terrier',
						 b'dandie_dinmont',
						 b'boston_bull',
						 b'miniature_schnauzer',
						 b'giant_schnauzer',
						 b'standard_schnauzer',
						 b'scotch_terrier',
						 b'tibetan_terrier',
						 b'silky_terrier',
						 b'coated_wheaten_terrier',
						 b'west_highland_white_terrier',
						 b'lhasa',
						 b'coated_retriever',
						 b'coated_retriever',
						 b'golden_retriever',
						 b'labrador_retriever',
						 b'chesapeake_bay_retriever',
						 b'haired_pointer',
						 b'vizsla',
						 b'english_setter',
						 b'irish_setter',
						 b'gordon_setter',
						 b'brittany_spaniel',
						 b'clumber',
						 b'english_springer',
						 b'welsh_springer_spaniel',
						 b'cocker_spaniel',
						 b'sussex_spaniel',
						 b'irish_water_spaniel',
						 b'kuvasz',
						 b'schipperke',
						 b'groenendael',
						 b'malinois',
						 b'briard',
						 b'kelpie',
						 b'komondor',
						 b'old_english_sheepdog',
						 b'shetland_sheepdog',
						 b'collie',
						 b'border_collie',
						 b'bouvier_des_flandres',
						 b'rottweiler',
						 b'german_shepherd',
						 b'doberman',
						 b'miniature_pinscher',
						 b'greater_swiss_mountain_dog',
						 b'bernese_mountain_dog',
						 b'appenzeller',
						 b'entlebucher',
						 b'boxer',
						 b'bull_mastiff',
						 b'tibetan_mastiff',
						 b'french_bulldog',
						 b'great_dane',
						 b'saint_bernard',
						 b'eskimo_dog',
						 b'malamute',
						 b'siberian_husky',
						 b'affenpinscher',
						 b'basenji',
						 b'pug',
						 b'leonberg',
						 b'newfoundland',
						 b'great_pyrenees',
						 b'samoyed',
						 b'pomeranian',
						 b'chow',
						 b'keeshond',
						 b'brabancon_griffon',
						 b'pembroke',
						 b'cardigan',
						 b'toy_poodle',
						 b'miniature_poodle',
						 b'standard_poodle',
						 b'mexican_hairless',
						 b'dingo',
						 b'dhole',
						 b'african_hunting_dog']

            document.getElementById("prediction").innerText = "The dog's breed is : " + labels_list[classId];

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
