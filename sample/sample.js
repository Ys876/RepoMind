// fake functions to be parsed by tree-sitter
// blehbleh
function purdue() {
    console.log("boilerup");
}

function login(user, pwd) {
    if ((user === "pete") && (pwd === "boilerUp123")) {
        console.log("success");
    } else {
        console.log("fail");
    }
}

function get_user() {
    return "pete";
}

function get_pwd() {
    return "boilerUp123";
}