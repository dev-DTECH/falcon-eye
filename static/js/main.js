function start_alert(querySelector) {
    let ele = document.querySelector(querySelector);
    var audio = new Audio('/alert.mp3');
    audio.play();
    ele.classList.add("alert")

    setTimeout(() => {
        ele.classList.remove("alert")
    }, 4000)
}

let query = [];
document.querySelector(".search-clip").addEventListener("submit", e => {
    e.preventDefault();
    query = document.querySelector(".search-text").value.split(" ")
    update_clips()
    // console.log(query)
    // console.log(clips)
})
function update_clip_url(url){
    let ele=document.querySelector(".clip-video")
    ele.src=`/clip/${url}`
    ele.play()
}
function update_clips() {
    let clips_ele = document.querySelector(".clip-container")
    clips_ele.innerHTML = ""
    for (let i = 0; i < clips.length; i++) {
        if (query.length === 0) {
            // console.log(clips[i])
            clips_ele.innerHTML += `<button class="clip" onclick="update_clip_url('${clips[i]}')">${clips[i]}</button>`
            continue;
        }
        for (let j = 0; j < query.length; j++) {
            if (clips[i].search(query[j]) !== -1) {
                clips_ele.innerHTML += `<button class="clip" onclick="update_clip_url('${clips[i]}')">${clips[i]}</button>`
                break;
            }
        }
    }
}