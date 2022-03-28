package com.example.myapplication;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Environment;
import android.view.View;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class Recorder extends AppCompatActivity {

    TextView textView;
    ImageButton btnStart, btnStop, btnRecordings;
    MediaRecorder mRecorder;
    MediaPlayer mPlayer;

    CountDownTimer timer;
    int secs = -1, min, hrs;
    String recFile;
    String audFile;
    public static final int PERMISSION_ALL = 0;
    String newFile;
    ImageButton btnHistory;

    @Override
    protected void onCreate (Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recorder);

        btnHistory = findViewById(R.id.historyClick);
        if (isMicrophonePresent()){
            getMicrophonePermission();
        }

        btnHistory.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent= new Intent(Recorder.this, HistoryActivity.class);
                startActivity(intent);
            }
        });

    }

    public void btnRecordPressed(View v){
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy_MM_dd_hh_mm_ss", Locale.CANADA);
        Date datePresent = new Date();

        recFile = "Recording_" + formatter.format(datePresent) + ".3gp";

        String filePath= Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC).getPath();
        File mainFolder=new File(filePath);
        if (!mainFolder.exists()){
            mainFolder.mkdir();
        }
        File folder= new File(filePath,"recorded");
        if (!folder.exists()){
            folder.mkdir();
        }

        newFile = folder+"/"+ recFile;

        try {

            mRecorder = new MediaRecorder();
            mRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            mRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
            mRecorder.setOutputFile(newFile);
            mRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
            mRecorder.prepare();
            mRecorder.start();

            showTimer();

            Toast.makeText(this,"Recording Started!",Toast.LENGTH_LONG).show();

        }
        catch (Exception e){
            e.printStackTrace();
        }


    }

    private void getRecordingFilePath(){
        Toast.makeText(this,"Nothing",Toast.LENGTH_LONG).show();
    }

    public void btnStopPressed(View v){
        mRecorder.stop();
        mRecorder.release();
        mRecorder = null;
        timer.cancel();
        secs = -1;
        min = 0;
        hrs = 0;
        textView.setText("00:00:00");

        Toast.makeText(this,"Recording Stoped",Toast.LENGTH_LONG).show();

    }

    public void btnHistoryPressed(View view) {
        /*
        Should display the page of the past recordings with a small audio player at the
        bottom of the screen. Please use recylerView.
        */


    }

    private boolean isMicrophonePresent(){
        if(this.getPackageManager().hasSystemFeature(PackageManager.FEATURE_MICROPHONE)){
            return true;
        }
        else {
            return false;
        }
    }

    private void getMicrophonePermission(){
         if(ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                 ==PackageManager.PERMISSION_DENIED){
             ActivityCompat.requestPermissions(this, new String[]
                     {Manifest.permission.RECORD_AUDIO},200);
        }
    }

    public void showTimer() {
        textView = (TextView) findViewById(R.id.text);
        timer = new CountDownTimer(Long.MAX_VALUE, 1000) {
            @Override
            public void onTick(long millisUntilFinished) {
                secs++;
                textView.setText(recorderTime());
            }
            public void onFinish() {

            }
        };
        timer.start();
    }
    public String recorderTime() {
        if (secs == 60) {
            min++;
            secs = 0;
        }
        if (min == 60) {
            hrs++;
            min = 0;
        }
        return String.format("%02d:%02d:%02d", hrs, min, secs);
    }


}

