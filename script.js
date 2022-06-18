let img_input = document.querySelector("#plant_leaf_image");
var upload_image = "";
let url = "http://localhost:8000/predict";
var disp_res = document.getElementById("disease_type");
var conf = document.getElementById("confidence");
img_input.addEventListener("change", function () {
  let imageFile = "";
  let reader = new FileReader();
  disp_res.innerText = "";
  conf.innerText = "";
  imageFile = img_input.value;
  reader.addEventListener("load", () => {
    upload_image = reader.result;
    if (reader.result) {
      document.querySelector(
        "#display_image"
      ).style.backgroundImage = `url(${upload_image})`;
    }
  });
  if (imageFile.length !== 0) reader.readAsDataURL(this.files[0]);
  else document.querySelector("#display_image").style.backgroundImage = "none";
});
