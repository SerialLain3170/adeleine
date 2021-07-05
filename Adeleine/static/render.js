function loadImages(i) {
    console.log(i);
    canvas = document.createElement('canvas');
    document.getElementById("lastElement").parentNode.insertBefore(canvas,document.getElementById("lastElement"));
    ctx = canvas.getContext('2d');

    if (i == 0){
      canvas.style.position = "absolute";
      canvas.style.left = "0px";
      canvas.style.backgroundColor = "gray";
    }
    if (i == 1){
      canvas.style.position = "absolute";
      canvas.style.left = "400px";
      canvas.style.backgroundColor = "gray";
    }
    if (i == 2){
      canvas.style.position = "absolute";
      canvas.style.left = "800px";
      canvas.style.backgroundColor = "gray";
    }

    var image = new Image();
    image.onload = (function () {
        const img_height = 300;
        const canvas_height = 350;
        const img_width = image.width * (img_height / image.height);
        canvas.width = canvas_height;
        canvas.height = canvas_height;
        const center = canvas_height / 2;
        const mid_height = img_height / 2;
        const mid_width = img_width / 2;
        const left = center - mid_width;
        const top = center - mid_height;
        ctx.drawImage(image, left, top, img_width, img_height);
    });

    if (i==0) {
      image.src = "/static/tmp_line.png";
    }
    if (i==1) {
      image.src = "/static/tmp_ref.png";
    }
    if (i==2) {
      image.src = "/static/tmp_result.png";
    }
}

function draw() {
  for (i=0;i<3;i++) {
    loadImages(i);
  }
}