#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QCoreApplication>
#include <QNetworkRequest>
#include <QJsonDocument>
#include <QJsonObject>

#include <iostream>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , networkManager(new QNetworkAccessManager(this))
{
    ui->setupUi(this);

    // Запускаем бэкенд в отдельном процессе
    #ifdef WIN32
    QString backendPath = QCoreApplication::applicationDirPath() + "/../backend.exe";
    #else
    QString backendPath = QCoreApplication::applicationDirPath() + "/backend";
    #endif

    std::cout << backendPath.toStdString();

    backendProcess.start(backendPath);
}

MainWindow::~MainWindow()
{
    backendProcess.kill();
    backendProcess.waitForFinished();
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    QNetworkRequest request(QUrl("http://127.0.0.1:5000/time"));
    QNetworkReply *reply = networkManager->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() { handleNetworkReply(reply); });
}

void MainWindow::handleNetworkReply(QNetworkReply *reply)
{
    if (reply->error() == QNetworkReply::NoError) {
        QByteArray data = reply->readAll();
        QJsonDocument doc = QJsonDocument::fromJson(data);
        QString time = doc.object().value("time").toString();
        ui->label->setText(time);
    }
    reply->deleteLater();
}
