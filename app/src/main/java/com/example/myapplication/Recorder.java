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
import android.widget.ImageView;
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
    ImageButton start, stop, recordings;
    MediaRecorder mediaRecorder;
    MediaPlayer mediaPlayer;

    CountDownTimer countDownTimer;
    int second = -1, minute, hour;
//    String filePath;
    String recordFile;
    String audioFile;
    public static final int PERMISSION_ALL = 0;
    String newFileName;
    ImageButton historyButton;

    @Override
    protected void onCreate (Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recorder);

        historyButton= findViewById(R.id.historyClick);
        if (isMicrophonePresent()){
            getMicrophonePermission();
        }

        historyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent= new Intent(Recorder.this, HistoryActivity.class);
                startActivity(intent);
            }
        });

    }

    public void btnRecordPressed(View v){
//        String recordPath = getExternalFilesDir("Voice Records").getAbsolutePath();
////        String recordPath = getExternalFilesDir("/").getAbsolutePath();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy_MM_dd_hh_mm_ss", Locale.CANADA);
        Date now = new Date();

        recordFile = "Recording_" + formatter.format(now) + ".3gp";

        String mainPath= Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC).getPath();
        File mainFolder=new File(mainPath);
        if (!mainFolder.exists()){
            mainFolder.mkdir();
        }
        File folder= new File(mainPath,"recorded");
        if (!folder.exists()){
            folder.mkdir();
        }

        newFileName = folder+"/"+recordFile;

//        filenameText.setText("Recording, File Name : " + recordFile);

        try {

            mediaRecorder = new MediaRecorder();
            mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
//            mediaRecorder.setOutputFile(getRecordingFilePath());
//            mediaRecorder.setOutputFile(recordPath + "/" + recordFile);
            mediaRecorder.setOutputFile(newFileName);
            mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
            mediaRecorder.prepare();
            mediaRecorder.start();

            showTimer();

            Toast.makeText(this,"Recording started",Toast.LENGTH_LONG).show();

        }
        catch (Exception e){
            e.printStackTrace();
        }


    }

    private void getRecordingFilePath(){
//        ContextWrapper contextWrapper = new ContextWrapper(getApplicationContext());
//        File md = contextWrapper.getExternalFilesDir(Environment.DIRECTORY_MUSIC);
//        File file = new File(md,"test"+".mp3");
//        return file.getPath();
        Toast.makeText(this,"Nothing",Toast.LENGTH_LONG).show();
//        return "A";
    }
    public void btnStopPressed(View v){
        mediaRecorder.stop();
        mediaRecorder.release();
        mediaRecorder = null;
        countDownTimer.cancel();
        second = -1;
        minute = 0;
        hour = 0;
        textView.setText("00:00:00");

        //creating content resolver and put the values
//        ContentValues values = new ContentValues();
//        values.put(MediaStore.Audio.Media.DATA, filePath);
//        values.put(MediaStore.Audio.Media.MIME_TYPE, "audio/3gpp");
//        values.put(MediaStore.Audio.Media.TITLE, audioFile);
//        //store audio recorder file in the external content uri
//        getContentResolver().insert(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, values);

        Toast.makeText(this,"Recording Stoped",Toast.LENGTH_LONG).show();

    }

//    public void btnPlayPressed(View v){
//
//        try {
//            mediaPlayer = new MediaPlayer();
////            mediaPlayer.setDataSource(getRecordingFilePath());
//            mediaPlayer.setDataSource(recordFile);
//            mediaPlayer.prepare();
//            mediaPlayer.start();
//            Toast.makeText(this,"Recording playing",Toast.LENGTH_LONG).show();
//
//        }
//        catch (Exception e){
//            e.printStackTrace();
//        }
//
//    }


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

/////////////////////////////
    public void showTimer() {
        textView = (TextView) findViewById(R.id.text);
        countDownTimer = new CountDownTimer(Long.MAX_VALUE, 1000) {
            @Override
            public void onTick(long millisUntilFinished) {
                second++;
                textView.setText(recorderTime());
            }
            public void onFinish() {

            }
        };
        countDownTimer.start();
    }
    public String recorderTime() {
        if (second == 60) {
            minute++;
            second = 0;
        }
        if (minute == 60) {
            hour++;
            minute = 0;
        }
        return String.format("%02d:%02d:%02d", hour, minute, second);
    }


}

