function getUrlVars() {
    var vars = [], hash;
    var hashes = window.location.href.slice(window.location.href.indexOf('?') + 1).split('&');
    for (var i = 0; i < hashes.length; i++) {
        hash = hashes[i].split('=');
        vars.push(hash[0]);
        vars[hash[0]] = hash[1];
    }
    return vars;
}

function extractHeader(data) {
    if (data['tableOrientation'] === 'HORIZONTAL') {
        return data['relation'].map(item => item[data['headerRowIndex']]);
    } else {
        return data['relation'][data['headerRowIndex']];
    }
}

function extractEntities(data) {
    relation = data['relation'];
    if (data['tableOrientation'] === 'HORIZONTAL') {
        const result = [];
        for (let i = 0; i < relation[0].length; i++) {
            if (i !== data['headerRowIndex']) {
                result.push(relation.map(item => item[i]));
            }
        }
        return result;
    } else {
        return relation.filter((item, idx) => idx !== data['headerRowIndex']);
    }
}

$(document).ready(function () {
    const vars = getUrlVars();
    if (vars.file) {
        $.getJSON(`http://127.0.0.1:3000/${vars.file}`, (data) => {
            $('#page-title').html(data.pageTitle);
            $('#title').html(data.title);
            $('#url').html(`<a href="${data.url}" target="_blank">${data.url}</a>`);
            extractHeader(data).forEach(label => {
                $('#table-header').append(`<th scope="col">${label}</th>`)
            });
            extractEntities(data).forEach(entity => {
                $('#table-body').append(`<tr>${entity.map(item => `<td>${item}</td>`).join('')}</tr>`)
            });
        });
    }
});
