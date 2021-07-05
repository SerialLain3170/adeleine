function uploadFile(event) {
	artType = this.artType;
	let filereader = new FileReader();
	filereader.readAsDataURL(event.target.files[0]);
	filereader.addEventListener('load', function(e) {

		let data = {image: e.target.result};

		const param = {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json; charset=utf-8'
			},
			body: JSON.stringify(data),
		};

		fetch('/reference/upload', param)
				.then((res) => {
					res.json().then(function(json) {
						if (artType == "line") {
							var element = document.querySelector('.line_result');
						} else if (artType == "ref") {
							var element = document.querySelector('.ref_result');
						}
						element.src = 'data:image/png;base64,' + json['data'];
						element.width = json['w'];
						element.height = json['h'];
				});
			});
		});
}

function refColorize(event) {
	const line_uri = document.querySelector('.line_result').src;
	const ref_uri = document.querySelector('.ref_result').src;
	const param  = {
		method: 'POST',
		headers: {
		'Content-Type': 'application/json; charset=utf-8'
		},
		body: JSON.stringify({image_line: line_uri, image_ref: ref_uri})
	};
	fetch('/reference', param)
		.then((res)=>{
			res.json().then(function(data) {
				var y_element = document.querySelector('.y_result');
				y_element.src = 'data:image/png;base64,' + data['data'];
				y_element.width = data['w'];
				y_element.height = data['h'];
			});
		});
}

window.addEventListener('DOMContentLoaded', function(){
	document.getElementById('line').addEventListener('change', {artType: "line", handleEvent: uploadFile});
	document.getElementById('ref').addEventListener('change', {artType: "ref", handleEvent: uploadFile});
	document.querySelector('.btn_reference').addEventListener('click', refColorize);
});
