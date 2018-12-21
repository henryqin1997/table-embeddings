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

function readTable(file) {
    $.getJSON(`http://127.0.0.1:3000/${file}`, data => {
        $('#page-title').html(data.pageTitle);
        $('#title').html(data.title);
        $('#url').html(`<a href="${data.url}" target="_blank">${data.url}</a>`);
        $('#raw').html(`<a href="${`http://127.0.0.1:3000/${file}`}" target="_blank">${file}</a>`);
        $('#table-header').empty();
        $('#table-body').empty();
        extractHeader(data).forEach(label => {
            $('#table-header').append(`<th scope="col">${label}</th>`)
        });
        extractEntities(data).forEach(entity => {
            $('#table-body').append(`<tr>${entity.map(item => `<td>${item}</td>`).join('')}</tr>`)
        });
    });
}

function addLink(key) {
    const index = key.indexOf(' ');
    const url = key.slice(0, index);
    return `<a href="${url}" target="_blank">${url}</a><br/>${key.slice(index)}`;
}

$(document).ready(function () {
    const vars = getUrlVars();

    if (vars.list) {
        $.getJSON(`http://127.0.0.1:3000/${vars.list}`, dict => {
            $('#radio-container').append(Object.keys(dict).map(key => `    
    <div class="radio">
        <label><input type="radio" name="domain-schema" value="${key}">${addLink(key)}</label>
    </div>`).join('\n'));

            $('input[type=radio]').change(function () {
                const file = dict[$(this).val()];
                readTable(`data/domain_schema_files/${file}`);
            });
        });
    } else {
        $('#left-section').remove();
        $('#right-section').css({'width': '100%', 'padding': '20px', 'box-sizing': 'border-box'});
    }

    if (vars.file) {
        readTable(vars.file);
    }
});
