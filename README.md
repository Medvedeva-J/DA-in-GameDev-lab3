# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил(а):
- Медведева Юлия Олеговна
- РИ-210940
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.

## Задача работы
Создать ML-агент и попробовать потренировать нейросеть, задача которой будет заключаться в управлении шаром. Задача шара заключается в том, чтобы оставаясь на плоскости находить кубик, смещающийся в заданном случайном диапазоне координат.

## Задание 1
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задач
Ход работы:
Создала новый проект на Unity, добавила MLAgent
![image](https://user-images.githubusercontent.com/62373163/197986740-cde2e68b-107b-4b38-aa18-67906bd7d5e0.png)

Написала серию команд для создания и активации нового ML-агента, а также для скачивания необходимых библиотек:

![image](https://user-images.githubusercontent.com/62373163/197987915-25345395-e270-4d5d-bcb7-5ad8b69159b5.png)
![image](https://user-images.githubusercontent.com/62373163/197988015-d1adad5e-0ef9-4c8c-aa57-73bf85f92e96.png)
![image](https://user-images.githubusercontent.com/62373163/197988103-255ad246-dbe7-467f-addd-bb2b0213df8f.png)
![image](https://user-images.githubusercontent.com/62373163/197988194-fdf78032-2251-489e-b65b-158301cf1236.png)
![image](https://user-images.githubusercontent.com/62373163/197988236-cfc14529-2c4e-4af1-a3bb-825353e736f9.png)

Создала плоскость, сферу и куб. К сфере подключила созданный С# скрипт и добавила компоненты Decision Requester, Behavior Parameters.

![image](https://user-images.githubusercontent.com/62373163/198014490-5fbd6372-a4d1-4191-8228-2fb8d4650c65.png)
![image](https://user-images.githubusercontent.com/62373163/198014631-a9889509-e273-414f-b3f6-17cccc7c4d4b.png)


```py
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;

    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;

    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }

    public float forceMultiplier = 10;

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}

```

В корень проекта добавила файл конфигурации нейронной сети, доступный в папке с файлами проекта. 
Запустила ml-агента и запустила сцену в Unity.
![image](https://user-images.githubusercontent.com/62373163/198018048-aa97926e-66e3-494b-87c5-78f09dc4960b.png)


```py

behaviors:
  RollerBall:
    trainer_type: ppo
    hyperparameters:
      batch_size: 10
      buffer_size: 100
      learning_rate: 3.0e-4
      beta: 5.0e-4
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000
    
```

## Задание 2
### Подробно опишите каждую строку файла конфигурации нейронной сети, доступного в папке с файлами проекта по ссылке. Самостоятельно найдите информацию о компонентах Decision Requester, Behavior Parameters, добавленных сфере.

```py

behaviors:
  RollerBall:
    trainer_type: ppo
    hyperparameters:
      batch_size: 10
      buffer_size: 100
      learning_rate: 3.0e-4
      beta: 5.0e-4
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000
    
```
1. trainer_type: ppo - тип обучения с поощрением.
2. hyperparameters: Гиперпараметр модели.
3. batch_size: Определяет количество информации, собираемой в ходе одного шага в градиентном спуске. Связан с параметром buffer_size, всегда должен быть его долей.
4. buffer_size: Количество опыта, который необходимо собрать перед обновлением модели политики. Соответствует тому, сколько опыта должно быть собрано, прежде чем мы приступим к какому-либо изучению или обновлению модели. Это должно быть в несколько раз больше, чем batch_size. Обычно больший размер буфера соответствует более стабильным обновлениям обучения.
5. learning_rate: Параметр задает величину каждого шага обновления градиентного спуска. Обычно это значение следует уменьшить, если обучение нестабильно, а вознаграждение не увеличивается последовательно.
- beta: Параметр задает степень регуляризации энтропии, и определяет, какой будет степень случайности в обновленной политике. Чем выше этот параметр, тем больше случайных действий будет выполнять модель. При оптимальном подборе параметра энтропия будет уменьшаться с увеличением среднего вознаграждения.
- epsilon: Влияет на то, насколько быстро политика может развиваться во время обучения. Соответствует допустимому порогу расхождения между старой и новой политиками при обновлении с градиентным спуском. Установка этого значения небольшим приведет к более стабильным обновлениям, но также замедлит процесс обучения.
- lambd: Обозначает то, насколько агент полагается на текущую оценку значения при расчете обновленной оценки значения. Чем выше значение, тем выше зависимость от фактических вознаграждений, полученных извне,что может привести к высокой дисперсии.
- num_epoch: Количество проходов, которые необходимо выполнить через буфер опыта при выполнении оптимизации градиентного спуска.Чем больше размер пакета, тем больше это допустимо сделать. Уменьшение этого параметра обеспечит более стабильные обновления за счет более медленного обучения.
- learning_rate_schedule : Определяет, как скорость обучения меняется с течением времени.
- network_settings: Параметры нейронной сети.
- normalize: К входным данным векторного наблюдения не будет применяться нормализация. Нормализация может быть полезна в случаях со сложными задачами непрерывного управления, но может быть вредна при более простых задачах дискретного управления.
- hidden_units: Соответствует количеству единиц в каждом полностью подключенном слое нейронной сети. Для простых задач, где правильным действием является простая комбинация входных данных наблюдения, это должно быть небольшим. Для задач, где действие представляет собой очень сложное взаимодействие между переменными наблюдения, это должно быть больше.
- num_layers: Определяет количество слоев нейронной сети. Для простых задач меньшее количество слоев, скорее всего, будет обучаться быстрее и эффективнее. Для более сложных задач управления может потребоваться больше уровней.
- reward_signals: Параметры вознаграждений.
- gamma: Параметр задает степень значимости будущих вознаграждений. Чем выше значение, тем больше агент должен действовать в настоящем, чтобы подготовиться к вознаграждению в отдаленном будущем.
- strength: Фактор, на который можно умножить вознаграждение, получаемое от окружающей среды. Типичные диапазоны будут варьироваться в зависимости от сигнала вознаграждения.
- max_steps: Максимальное количество проходов.
- time_horizon: Временной отрезок.
- summary_freq: Суммированная частота.
- DecisionRequester запрашивает принятие решения на основе сделанных наблюдений через определенные промежутки времени. Иными словами, DecisionRequester определяет, сколько шагов Академии должно быть выполнено, прежде чем будет запрошено решение.
- BehaviorParameters определяет, сколько наблюдений мы принимаем и какую форму будут принимать выводимые действия.
- BehaviorType имеет три варианта: эвристический (heuristic), по умолчанию (default) и вывод (inference). При установке значения по умолчанию, если нейронная сеть была сгенерирована, агент будет выполнять Inference, поскольку он использует нейронную сеть для принятия решений. Когда нейронная сеть не предусмотрена, она будет использовать эвристику. Эвристический метод можно рассматривать как традиционный подход к искусственному интеллекту, при котором программист вводит все возможные команды непосредственно в объект. Если установлено значение Heuristic only, агент будет запускать все, что находится в эвристическом методе.


## Задание 3
### Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и впервом задании, случайно изменять кооринаты на плоскости. 

- Код нового С#-скрипта:

```py

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public GameObject Target;
    public GameObject Target1;
    private bool touch_target;
    private bool touch_target1;

    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.transform.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
        Target1.transform.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
        Target.SetActive(true);
        Target1.SetActive(true);
        touch_target = false;
        touch_target1 = false;
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.transform.localPosition);
        sensor.AddObservation(Target1.transform.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(touch_target);
        sensor.AddObservation(touch_target1);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }

    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.transform.localPosition);
        float distanceToTarget1 = Vector3.Distance(this.transform.localPosition, Target1.transform.localPosition);

        if (!touch_target & distanceToTarget < 1.42f)
        {
            touch_target = true;
            Target.SetActive(false);
        }

        if (!touch_target1 & distanceToTarget1 < 1.42f)
        {
            touch_target1 = true;
            Target1.SetActive(false);

        }

        if (touch_target & touch_target1)
        {
            SetReward(1.0f);
            EndEpisode();
        }

        else if (this.transform.localPosition.y < 0)
        {
            SetReward(-0.5f);
            EndEpisode();
        }
    }
}

```

-ДОПОЛНИТЬ


## Выводы

В ходе выполонения данной лабораторной работы я ознакомилась с основами использования ML агента для оптимизации траектории движения интеллектуального агента.

Игровой баланс - соблюдение равновесия между различными экономическими показателями в игре. Это включает в себя настройку сложности, условий выигрыша / проигрыша, игровых состояний, баланса экономики и так далее, чтобы работать в тандеме друг с другом. В таком случае системы машинного обучения можно использовать для создания интеллектуальных агентов с оптимальной сложностью прохождения для игрока или подбора коэффициентов в экономической системе со множеством параметров.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
