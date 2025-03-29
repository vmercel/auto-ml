$(document).ready(function() {
    let pipeline = [];
    const pipelineArea = $('#pipeline')[0];

    // Debug jQuery UI
    console.log("jQuery version:", $.fn.jquery);
    console.log("jQuery UI loaded:", typeof $.ui !== 'undefined');
    console.log("Sortable available:", typeof $.fn.sortable !== 'undefined');

    if (typeof $.fn.sortable === 'undefined') {
        alert("jQuery UI Sortable not loaded. Check script includes.");
        return;
    }

    // Make pipeline sortable
    $("#pipeline").sortable({
        placeholder: "ui-state-highlight",
        update: function() {
            pipeline = [];
            $('.pipeline-item').each(function() {
                const type = $(this).data('type');
                const params = $(this).find('input').val() ? parseParams($(this).find('input').val()) : {};
                pipeline.push({ type: type, params: params });
            });
        }
    });

    // Default parameters defined client-side
    const IMPUTERS = {
        'MeanImputer': {'strategy': 'mean'},
        'MedianImputer': {'strategy': 'median'},
        'KNNImputer': {'n_neighbors': 5}
    };

    const ENCODERS = {
        'LabelEncoder': {'target': 'false'},
        'OneHotEncoder': {'sparse_output': 'false', 'drop': 'first'}  // 'drop' as string
    };

    const PREPROCESSORS = {
        'StandardScaler': {},
        'MinMaxScaler': {'feature_range': '0,1'}
    };

    const FEATURE_SELECTORS = {
        'SelectKBest': {'k': 10, 'score_func': 'f_classif'}
    };

    const MODELS = {
        'classification': {
            'LogisticRegression': {'C': 1.0, 'max_iter': 1000},
            'DecisionTree': {'max_depth': 5},
            'RandomForest': {'n_estimators': 100},
            'NeuralNetwork': {'layers': '64,32', 'dropout': 0.2, 'epochs': 20}
        },
        'regression': {
            'LinearRegression': {},
            'DecisionTree': {'max_depth': 5},
            'RandomForest': {'n_estimators': 100},
            'NeuralNetwork': {'layers': '64,32', 'dropout': 0.2, 'epochs': 20}
        }
    };

    // Sidebar collapsible menu
    $('.menu-item h3').click(function() {
        $(this).next('.sub-items').slideToggle(200);
    });

    // Drag-and-drop from sidebar to pipeline
    $('.component').on('dragstart', function(e) {
        e.originalEvent.dataTransfer.setData('text/plain', $(this).data('type'));
        $(this).addClass('dragging');
    }).on('dragend', function() {
        $(this).removeClass('dragging');
    });

    pipelineArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        $(this).addClass('drag-over');
    });

    pipelineArea.addEventListener('dragleave', function() {
        $(this).removeClass('drag-over');
    });

    pipelineArea.addEventListener('drop', function(e) {
        e.preventDefault();
        $(this).removeClass('drag-over');
        const type = e.dataTransfer.getData('text');
        const defaultParams = getDefaultParams(type);
        const paramString = Object.entries(defaultParams).map(([k, v]) => `${k}=${v}`).join(', ');
        const item = $(`
            <div class="pipeline-item" data-type="${type}">
                <span>${type}</span>
                <input type="text" value="${paramString}" placeholder="e.g., C=1.0, layers=64,32">
                <button class="delete-btn">Delete</button>
            </div>
        `);
        $(this).append(item);
        pipeline.push({ type: type, params: defaultParams });

        // Delete functionality
        item.find('.delete-btn').click(function() {
            item.remove();
            pipeline = pipeline.filter(p => p.type !== type || JSON.stringify(p.params) !== JSON.stringify(defaultParams));
        });
    });

    // Upload Data
    $('#uploadButton').click(function() {
        const file = $('#dataFile')[0].files[0];
        if (!file) {
            alert('Please select a file');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);

        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                let html = '<h3>Data Summary</h3>';
                html += `<p>Shape: ${response.summary.shape[0]} rows, ${response.summary.shape[1]} columns</p>`;
                html += '<h4>First 5 Rows:</h4><table>';
                html += '<tr>' + Object.keys(response.summary.head).map(col => `<th>${col}</th>`).join('') + '</tr>';
                for (let i = 0; i < 5; i++) {
                    html += '<tr>' + Object.keys(response.summary.head).map(col => `<td>${response.summary.head[col][i] || ''}</td>`).join('') + '</tr>';
                }
                html += '</table>';
                html += '<h4>Field Types:</h4><ul>' + Object.entries(response.summary.dtypes).map(([col, type]) => `<li>${col}: ${type}</li>`).join('') + '</ul>';
                html += '<p>Numeric Features: ' + response.summary.numeric.join(', ') + '</p>';
                html += '<p>Categorical Features: ' + response.summary.categorical.join(', ') + '</p>';
                html += '<h4>Missing Values:</h4><ul>' + Object.entries(response.summary.missing).map(([col, count]) => `<li>${col}: ${count}</li>`).join('') + '</ul>';
                html += '<h4>Summary Statistics:</h4><table>';
                html += '<tr><th></th>' + Object.keys(response.summary.stats).map(col => `<th>${col}</th>`).join('') + '</tr>';
                for (let stat in response.summary.stats[Object.keys(response.summary.stats)[0]]) {
                    html += `<tr><td>${stat}</td>` + Object.keys(response.summary.stats).map(col => `<td>${response.summary.stats[col][stat]}</td>`).join('') + '</tr>';
                }
                html += '</table>';
                $('#dataSummary').html(html);
                if (response.heatmap) {
                    $('#heatmap').html(`<img src="data:image/png;base64,${response.heatmap}" alt="Correlation Heatmap">`);
                } else {
                    $('#heatmap').html('<p>No numeric columns available for correlation heatmap.</p>');
                }

                // Populate target feature dropdown
                $.getJSON('/get_columns', function(data) {
                    console.log("Columns from /get_columns:", data.columns);
                    let targetHtml = '';
                    data.columns.forEach(col => {
                        targetHtml += `<option value="${col}">${col}</option>`;
                    });
                    $('#targetFeature').html(targetHtml);
                }).fail(function(xhr) {
                    console.error("Error fetching columns:", xhr.responseJSON);
                    alert('Error fetching column names: ' + (xhr.responseJSON ? xhr.responseJSON.error : 'Unknown error'));
                });
            },
            error: function(xhr) {
                alert('Error: ' + xhr.responseJSON.error);
            }
        });
    });

    // Set Task
    $('#setTaskButton').click(function() {
        const taskType = $('#taskType').val();
        const targets = $('#targetFeature').val();
        if (!targets || targets.length === 0) {
            alert('Please select at least one target feature');
            return;
        }
        $.ajax({
            url: '/set_task',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ task_type: taskType, targets: targets }),
            success: function(response) {
                console.log("Task set response:", response);
                loadModels(taskType);
            },
            error: function(xhr) {
                console.error("Error setting task:", xhr.responseJSON);
                alert('Error: ' + xhr.responseJSON.error);
            }
        });
    });

    // Train Pipeline
    $('#trainButton').click(function() {
        pipeline = [];
        $('.pipeline-item').each(function() {
            const type = $(this).data('type');
            const params = $(this).find('input').val() ? parseParams($(this).find('input').val()) : {};
            pipeline.push({ type: type, params: params });
        });

        $.ajax({
            url: '/train',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ pipeline: pipeline }),
            success: function(response) {
                const metrics = response.metrics;
                let metricsHtml = '<h3>Metrics</h3><ul>';
                for (let metric in metrics) {
                    metricsHtml += `<li>${metric}: ${metrics[metric].toFixed(4)}</li>`;
                }
                metricsHtml += '</ul>';
                $('#metrics').html(metricsHtml);
                Plotly.newPlot('plot', JSON.parse(response.plot));
                $('#results').show();
            },
            error: function(xhr) {
                alert('Error: ' + xhr.responseJSON.error);
            }
        });
    });

    // Save Pipeline
    $('#savePipeline').click(function() {
        pipeline = [];
        $('.pipeline-item').each(function() {
            const type = $(this).data('type');
            const params = $(this).find('input').val() ? parseParams($(this).find('input').val()) : {};
            pipeline.push({ type: type, params: params });
        });
        $.ajax({
            url: '/save_pipeline',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ pipeline: pipeline }),
            success: function(response) {
                alert(response.message);
            }
        });
    });

    // Load Pipeline
    $('#loadPipeline').click(function() {
        $.getJSON('/load_pipeline', function(response) {
            pipeline = response.pipeline;
            $('#pipeline').empty();
            pipeline.forEach(step => {
                const paramString = Object.entries(step.params).map(([k, v]) => `${k}=${v}`).join(', ');
                const item = $(`
                    <div class="pipeline-item" data-type="${step.type}">
                        <span>${step.type}</span>
                        <input type="text" value="${paramString}" placeholder="e.g., C=1.0, layers=64,32">
                        <button class="delete-btn">Delete</button>
                    </div>
                `);
                $('#pipeline').append(item);
                item.find('.delete-btn').click(function() {
                    item.remove();
                    pipeline = pipeline.filter(p => p.type !== step.type || JSON.stringify(p.params) !== JSON.stringify(step.params));
                });
            });
            alert('Pipeline loaded');
        }).fail(function(xhr) {
            alert('Error: ' + xhr.responseJSON.error);
        });
    });

    function loadModels(taskType) {
        const models = taskType === 'classification' ? [
            {type: 'LogisticRegression', tooltip: 'Linear model for classification'},
            {type: 'DecisionTree', tooltip: 'Tree-based classifier'},
            {type: 'RandomForest', tooltip: 'Ensemble of decision trees'},
            {type: 'NeuralNetwork', tooltip: 'Deep learning model'}
        ] : [
            {type: 'LinearRegression', tooltip: 'Linear model for regression'},
            {type: 'DecisionTree', tooltip: 'Tree-based regressor'},
            {type: 'RandomForest', tooltip: 'Ensemble of decision trees'},
            {type: 'NeuralNetwork', tooltip: 'Deep learning model'}
        ];
        
        let html = '';
        models.forEach(model => {
            html += `<div class="component" draggable="true" data-type="${model.type}">${model.type}</div>`;
        });
        $('#model-sub-items').html(html);

        $('#model-sub-items .component').on('dragstart', function(e) {
            e.originalEvent.dataTransfer.setData('text/plain', $(this).data('type'));
            $(this).addClass('dragging');
        }).on('dragend', function() {
            $(this).removeClass('dragging');
        });
    }

    function getDefaultParams(type) {
        if (type in IMPUTERS) return IMPUTERS[type];
        if (type in ENCODERS) return ENCODERS[type];
        if (type in PREPROCESSORS) return PREPROCESSORS[type];
        if (type in FEATURE_SELECTORS) return FEATURE_SELECTORS[type];
        const taskType = $('#taskType').val();
        if (taskType && type in MODELS[taskType]) return MODELS[taskType][type];
        return {};
    }

    function parseParams(input) {
        const params = {};
        if (!input) return params;
        input.split(',').forEach(param => {
            const [key, value] = param.split('=');
            if (key.trim() === 'layers') {
                params[key.trim()] = value.split(',').map(Number);
            } else if (key.trim() === 'target' || key.trim() === 'sparse_output') {
                params[key.trim()] = value.trim() === 'true';
            } else {
                params[key.trim()] = (key.trim() === 'drop') ? value.trim() : (parseFloat(value) || value);
            }
        });
        return params;
    }
});